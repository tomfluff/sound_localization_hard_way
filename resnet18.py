import torchvision.models as models
import torch
import tensorflow as tf
import numpy as np
import PIL, cv2

# === RESNET ===

def BasicBlock(inputs, num_channels, kernel_size, num_blocks, skip_blocks, name):
    """Basic residual block"""
    x = inputs

    for i in range(num_blocks):
        if i not in skip_blocks:
            x1 = ConvNormRelu(x, num_channels, kernel_size, strides=[1,1], name=name + '.'+str(i))
            x = tf.keras.layers.Add()([x, x1])
            x = tf.keras.layers.Activation('relu')(x)
    return x

def BasicBlockDown(inputs, num_channels, kernel_size, name):
    """Residual block with strided downsampling"""
    x = inputs
    x1 = ConvNormRelu(x, num_channels, kernel_size, strides=[2,1], name=name+'.0')
    x = tf.keras.layers.Conv2D(num_channels, kernel_size=1, strides=2, padding='same', activation='linear', use_bias=False, name=name+'.0.downsample.0')(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.1, epsilon=1e-5, name=name+'.0.downsample.1')(x)
    x = tf.keras.layers.Add()([x, x1])
    x = tf.keras.layers.Activation('relu')(x)
    return x

def ConvNormRelu(x, num_channels, kernel_size, strides, name):
    """Layer consisting of 2 consecutive batch normalizations with 1 first relu"""
    if strides[0] == 2:
        x = tf.keras.layers.ZeroPadding2D(padding=(1,1), name=name+'.pad')(x)
        x = tf.keras.layers.Conv2D(num_channels, kernel_size, strides[0], padding='valid', activation='linear', use_bias=False, name=name+'.conv1')(x)
    else:
        x = tf.keras.layers.Conv2D(num_channels, kernel_size, strides[0], padding='same', activation='linear',  use_bias=False, name=name+'.conv1')(x)      
    x = tf.keras.layers.BatchNormalization(momentum=0.1, epsilon=1e-5, name=name+'.bn1')(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Conv2D(num_channels, kernel_size, strides[1], padding='same', activation='linear', use_bias=False, name=name+'.conv2')(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.1, epsilon=1e-5, name=name+'.bn2')(x)
    return x

def ResNet18(inputs):
    x = tf.keras.layers.ZeroPadding2D(padding=(3,3), name='pad')(inputs)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=7, strides=2, padding='valid', activation='linear', use_bias=False, name='conv1')(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.1, epsilon=1e-5, name='bn1')(x)
    x = tf.keras.layers.Activation('relu', name='relu')(x)
    x = tf.keras.layers.ZeroPadding2D(padding=(1,1), name='pad1')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='valid', name='maxpool')(x)

    x = BasicBlock(x, num_channels=64, kernel_size=3, num_blocks=2, skip_blocks=[], name='layer1')

    x = BasicBlockDown(x, num_channels=128, kernel_size=3, name='layer2')
    x = BasicBlock(x, num_channels=128, kernel_size=3, num_blocks=2, skip_blocks=[0], name='layer2')

    x = BasicBlockDown(x, num_channels=256, kernel_size=3, name='layer3')
    x = BasicBlock(x, num_channels=256, kernel_size=3, num_blocks=2, skip_blocks=[0], name='layer3')

    x = BasicBlockDown(x, num_channels=512, kernel_size=3, name='layer4')
    x = BasicBlock(x, num_channels=512, kernel_size=3, num_blocks=2, skip_blocks=[0], name='layer4')
    
    x = tf.keras.layers.GlobalAveragePooling2D(name='avgpool')(x)
    x = tf.keras.layers.Dense(units=1000, use_bias=True, activation='linear', name='fc')(x)

    return x

# === IMPORTING ===

def torch_layer_names(resnet_torch):
    torch_layer_names = []
    for name, module in resnet_torch.named_modules():
        torch_layer_names.append(name)
    return torch_layer_names

def import_torch_imagenet_weights(resnet_torch, torch_layer_names, ResNet18):
    inputs_i = tf.keras.Input((None, None, 3))
    resnet_tf_i = ResNet18(inputs_i)
    model = tf.keras.Model(inputs_i, resnet_tf_i)

    tf_layer_names = [layer.name for layer in model.layers]
    tf_layer_names = [layer for layer in tf_layer_names if layer in torch_layer_names]

    for layer in tf_layer_names:
        if 'conv' in layer:
            tf_conv = model.get_layer(layer)
            weights = resnet_torch.state_dict()[layer+'.weight'].numpy()
            weights_list = [weights.transpose((2, 3, 1, 0))]
            if len(tf_conv.weights) == 2:
                bias = resnet_torch.state_dict()[layer+'.bias'].numpy()
                weights_list.append(bias)
            tf_conv.set_weights(weights_list)
        elif 'bn' in layer:
            tf_bn = model.get_layer(layer)
            gamma = resnet_torch.state_dict()[layer+'.weight'].numpy()
            beta = resnet_torch.state_dict()[layer+'.bias'].numpy()
            mean = resnet_torch.state_dict()[layer+'.running_mean'].numpy()
            var = resnet_torch.state_dict()[layer+'.running_var'].numpy()
            bn_list = [gamma, beta, mean, var]
            tf_bn.set_weights(bn_list)
        elif 'downsample.0' in layer:
            tf_downsample = model.get_layer(layer)
            weights = resnet_torch.state_dict()[layer+'.weight'].numpy()
            weights_list = [weights.transpose((2, 3, 1, 0))]
            if len(tf_downsample.weights) == 2:
                bias = resnet_torch.state_dict()[layer+'.bias'].numpy()
                weights_list.append(bias)
            tf_downsample.set_weights(weights_list)
        elif 'downsample.1' in layer:
            tf_downsample = model.get_layer(layer)
            gamma = resnet_torch.state_dict()[layer+'.weight'].numpy()
            beta = resnet_torch.state_dict()[layer+'.bias'].numpy()
            mean = resnet_torch.state_dict()[layer+'.running_mean'].numpy()
            var = resnet_torch.state_dict()[layer+'.running_var'].numpy()
            bn_list = [gamma, beta, mean, var] # [gamma, beta, mean, var]
            tf_downsample.set_weights(bn_list)
        elif 'fc' in layer:
            tf_fc = model.get_layer(layer)
            weights = resnet_torch.state_dict()[layer+'.weight'].numpy() 
            weights_list = [weights.transpose((1, 0))]
            if len(tf_fc.weights) == 2:
                bias = resnet_torch.state_dict()[layer+'.bias'].numpy()
                weights_list.append(bias)
            tf_fc.set_weights(weights_list)
        else:
            print('No parameters found for {}'.format(layer))
    return model

def compare_keras_to_torch_resnet(resnet_torch, model):
    img = np.expand_dims(cv2.resize(src=np.array(PIL.Image.open('cat.png', 'r')),dsize=(224,224,),interpolation=cv2.INTER_CUBIC), 0).astype(np.float32)
    print(f'Original image dim: {img.shape}')
    img_torch = torch.tensor(img.transpose((0, 3, 1, 2)))
    print(f'Image torch dim: {img_torch.shape}')
    tf_output = model.predict(img)
    resnet_torch.eval()
    torch_output = resnet_torch(img_torch)

    max_diff = np.max(np.abs(tf_output - torch_output.detach().numpy()))
    print('Max difference in fully connected layer :{}'.format(max_diff))

# === MAIN ===
if __name__ == "__main__":
    # get the pytorch resnet
    resnet_torch = models.resnet18(pretrained=True)
    resnet_torch.state_dict
    # get and print pytorch layer names
    torch_layer_names = torch_layer_names(resnet_torch)
    print('PyTorch ResNet18 Layers: {}'.format(torch_layer_names))
    # generate full Keras resnet18 pretrained model
    model = import_torch_imagenet_weights(resnet_torch, torch_layer_names, ResNet18)

    # compare the models to check accuracy
    compare_keras_to_torch_resnet(resnet_torch, model)
    # save the resnet pretrained weights
    model.save_weights('imagenet.resnet18.top.h5')


