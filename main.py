import csv,os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import keras.api._v2.keras as keras # workaround: stackoverflow.com/questions/71000250
from matplotlib import pyplot as plt
import numpy as np
import cv2

# from dataloader import AudioVideoDataset
from datagen import AudioVideoDataGenerator
from model import LocalSoundModel

FETCH_SIZE = 270
BATCH_SIZE = 6
EPOCH = 4
TRAIN = False

def run_model(st=0, en=1):
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    
    # model
    inputs = [
        keras.Input(shape=(448,448,3), name='vision_in'),
        keras.Input(shape=(257,925,1), name='audio_in')
        ]
    model = LocalSoundModel(inputs, batch_size=BATCH_SIZE)
    # create an early stopping callback
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-7), 
        run_eagerly=False
        )
    
    # get data generator
    av_dataset = AudioVideoDataGenerator(data_fetch_size=FETCH_SIZE)
    if TRAIN:
        model.build([(BATCH_SIZE,448,448,3),(BATCH_SIZE,257,925)])
        if st > 0:
            model.load_weights(f'localsound.{st-1}.h5')
        
        loss_log = []
        for i in range(st,en):
            # train the model
            in_tensors = av_dataset.dataset(i)
            history = model.fit(
                x=in_tensors, 
                batch_size=BATCH_SIZE, 
                epochs=EPOCH, 
                shuffle=True
                )
            loss_log.extend([history.history["loss"]])
            model.finalize_state()
            model.save_weights(f'localsound.{i}.h5')
            in_tensors = None
        with open('loss_data.csv', 'a+', newline='') as f:
            csv_w = csv.writer(f)
            csv_w.writerows(loss_log)
    else:
        model.build([(BATCH_SIZE,448,448,3),(BATCH_SIZE,257,925)])
        model.load_weights('localsound.8.h5')
        
        av_test = AudioVideoDataGenerator(csv_path='test.csv', data_fetch_size=24)
        preds = model.predict(
            x=av_test.dataset(0),
            batch_size=BATCH_SIZE
            )
        
        k = 0
        for i in range(len(preds[0])):
            f, axarr = plt.subplots(1,4)
            hm = preds[1][i,0]
            pos = preds[2][i,0]
            img = keras.preprocessing.image.smart_resize(preds[3][i],(224,224))
            axarr[0].imshow(keras.preprocessing.image.array_to_img(img))
            axarr[1].matshow(cv2.resize(np.array(hm), dsize=(224, 224), interpolation=cv2.INTER_LINEAR))
            axarr[2].matshow(cv2.resize(np.array(pos), dsize=(224, 224), interpolation=cv2.INTER_LINEAR))
            axarr[3].imshow(keras.preprocessing.image.array_to_img(img*(cv2.resize(np.array(pos), dsize=(224, 224), interpolation=cv2.INTER_LINEAR)[:,:,None])))
            for ax in axarr:
                ax.axes.xaxis.set_visible(False)
                ax.axes.yaxis.set_visible(False)
            f.savefig(f'batch_8/img_{k}', bbox_inches='tight')
            k += 1
    
if __name__ == "__main__":
    run_model()
