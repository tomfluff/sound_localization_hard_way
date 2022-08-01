import tensorflow as tf
import keras.api._v2.keras as keras # workaround: stackoverflow.com/questions/71000250

DEBUG = False

# === RESNET-18 IMPLEMENTATION ===
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
    # Note: "TOP" part of the resnet is not included
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

    return x

# === MAIN MODEL IMPLEMENTATION ===

class LocalSoundModel(keras.Model):
    def __init__(self, inputs, batch_size=32) -> None:
        super(LocalSoundModel, self).__init__()
        self.batch_size = batch_size
        self.mask = 1.0 - 100.0 * tf.eye(self.batch_size, self.batch_size)
        if DEBUG: print(f'mask={self.mask}')
        self.loss_tracker = keras.metrics.Mean(name="loss")
        
        self.in_frame = inputs[0]
        self.in_audio = inputs[1]
        # frame (image) resnet [pretrained on imagenet]
        self.frame_model = tf.keras.Model(self.in_frame, ResNet18(self.in_frame), name='vision_resnet18')
        self.frame_model.load_weights('imagenet.resnet18.h5')
        # audio resnet
        self.audio_model = tf.keras.Model(self.in_audio,ResNet18(self.in_audio), name='audio_resnet18')
        # other layers
        self.audio_pool2d = keras.layers.GlobalAveragePooling2D(name='audio_pool2d')
        self.frame_norm = keras.layers.Normalization(mean=[0.485, 0.456, 0.406], variance=[0.229, 0.224, 0.225],name='vision_norm')
        
        self.eps_pos = 0.65
        self.eps_neg = 0.4
        self.tau = 0.03
        self.temperture = 0.07
    
    @property
    def metrics(self):
        return [self.loss_tracker]
    
    def __calc_loss(self, data):
        Pi = data[:,0:1]
        Ni = tf.reduce_sum(data[:,1:],axis=1)
        if DEBUG: print(f'Pi={Pi}, Ni={Ni}')
        Pi_e = tf.math.exp(Pi)
        Ni_e = tf.math.exp(Ni)
        if DEBUG: print(f'Pi_e={Pi_e}, Ni_e={Ni_e}')
        
        loss = (-1.0 / self.batch_size) * tf.reduce_sum(tf.math.log(Pi_e / (Pi_e + Ni_e)))
        if DEBUG: print(f'loss={loss.shape}, {loss}')
        return loss
    
    def __calc_reinforcement(self, frame, audio):
        # === LOSS CALCULATION ===
        # join together (audio and frame are already normalized)
        # frame = tf.transpose(frame, perm=[0,3,1,2])
        S_ii = tf.expand_dims(tf.einsum('nqac,nchw->nqa', frame, tf.expand_dims(tf.expand_dims(audio,2),3)), 1)
        S_ij = tf.einsum('nqac,ckhw->nkqa', frame, tf.expand_dims(tf.expand_dims(tf.transpose(audio),2),3))
        if DEBUG: print(f'S_ii={S_ii.shape} ({tf.reduce_mean(S_ii)}), S_ij={S_ij.shape} ({tf.reduce_mean(S_ij)})')
        
        # trimap generation
        m_ip = tf.sigmoid((S_ii-self.eps_pos)/self.tau)
        m_in = tf.sigmoid((S_ii-self.eps_neg)/self.tau)
        neg = 1 - m_in
        if DEBUG: print(f'm_ip={m_ip.shape} ({tf.reduce_mean(m_ip)}), m_in={m_in.shape} ({tf.reduce_mean(m_in)}), neg={neg.shape} ({tf.reduce_mean(neg)})')
        
        # Positive (Pi)
        Pi = tf.reduce_sum(tf.reshape((m_ip*S_ii), (*S_ii.shape[:2],-1)),axis=-1) / tf.reduce_sum(tf.reshape(m_ip,(*m_ip.shape[:2],-1)),axis=-1)
        # Negative (Ni)
        # easy neragives
        n_all = tf.sigmoid((S_ij-self.eps_pos)/self.tau)
        if DEBUG: print(f'n_all={n_all.shape}')
        Ni_easy = (tf.reduce_sum(tf.reshape(n_all*S_ij,(*S_ij.shape[:2],-1)),axis=-1) / tf.reduce_sum(tf.reshape(n_all,(*n_all.shape[:2],-1)),axis=-1)) * self.mask
        # hard negatives
        Ni_hard = tf.reduce_sum(tf.reshape((neg*S_ii), (*S_ii.shape[:2],-1)),axis=-1) / tf.reduce_sum(tf.reshape(neg,(*neg.shape[:2],-1)),axis=-1)
        if DEBUG: print(f'Pi={Pi.shape} ({tf.reduce_sum(Pi)}), Ni_easy={Ni_easy.shape} ({tf.reduce_sum(Ni_easy,axis=1)}), Ni_hard={Ni_hard.shape} ({tf.reduce_sum(Ni_hard)})')
        
        data = tf.concat([Pi,Ni_hard,Ni_easy],axis=1)*self.temperture
        if DEBUG: print(f'data={data.shape} {data}')
        
        return data, S_ii, m_ip
    
    def call(self, inputs, training=None, mask=None):
        frame_in, audio_in = inputs
        # frame
        if DEBUG: print(f'frame_in={frame_in[0,50,50]}')
        frame = self.frame_norm(frame_in / 255.0)
        if DEBUG: print(f'frame_norm={frame[0,50,50]}')
        frame = self.frame_model(frame)
        if DEBUG: print(f'resnet(frame)={frame.shape} ({frame[0,0,0,0:5]})')
            
        # audio
        if DEBUG: print(f'audio_in={audio_in[0,50,0:2]}')
        audio = self.audio_model(audio_in)
        if DEBUG: print(f'resnet(audio)={audio.shape}')
        audio = self.audio_pool2d(audio)
        if DEBUG: print(f'pool2d(audio)={audio.shape} ({audio[0,0:5]})')
            
        frame = tf.math.l2_normalize(frame,axis=3)
        if DEBUG: print(f'normal(frame)={frame.shape} ({frame[0,0,0,0:5]})')
        audio = tf.math.l2_normalize(audio,axis=1)
        if DEBUG: print(f'normal(audio)={audio.shape} ({audio[0,0:5]})')
        
        data, heatmap, pos_heatmap = self.__calc_reinforcement(frame, audio)
        return data, heatmap, pos_heatmap, frame_in
    
    def train_step(self, data_in):
        # wroks on a batch of data ( data=tuple(X,) ; X=tuple(frame, audio) )
        frame_in, audio_in = data_in[0]
        # === FORWARD PASS ===
        with tf.GradientTape() as tape:
            # forward pass on data
            data, _, _, _ = self.call((frame_in, audio_in))
            # calculation of loss
            loss = self.__calc_loss(data)
        
        learnable_params = (
            self.frame_model.trainable_variables + self.audio_model.trainable_variables
        )
        gradients = tape.gradient(loss, learnable_params)
        self.optimizer.apply_gradients(zip(gradients, learnable_params))

        # monitor loss
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}