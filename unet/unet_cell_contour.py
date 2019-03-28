
from keras.layers import Input, Conv2D, MaxPooling2D, concatenate, UpSampling2D
from keras.models import Model
from keras.optimizers import Adam
import keras
from random import shuffle, randint
import numpy as np
import os
import tensorflow as tf
import glob
from keras import backend as K
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import rotate
import pandas as pd
import SimpleITK as sitk
from skimage.measure import label
from scipy.ndimage.morphology import binary_closing
import time



GPU = 0
os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU)



def dice_coef(y_true, y_pred):
    smooth = 1
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def get_unet():
    inputs = Input((200, 200, 1))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])

    return model



def load_data(imgs_f, idxs_f):
    ret_data = np.zeros((len(idxs_f), 200, 200, 1))
    for i, idx in enumerate(idxs_f):
        ret_data[i, :,:,0] = imgs_f[:,:, idx]
    return ret_data


def getLargestCC(segmentation):
    if np.max(segmentation) == 0:
        return segmentation
    labels = label(segmentation)
    largestCC = (labels == np.argmax(np.bincount(labels.flat)[1::])+1)
    return largestCC


if __name__ == "__main__":

    mode = "Test"  # "Train" or "Test" or "Debug" or "Create_Masks_All_Images_Unet"


    data_directory = "deep_learning/unet-2D/"

    output_directory = data_directory + "/training"
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    m_batch = 25

    print("Loading sitk images")
    imgs = sitk.GetArrayFromImage(sitk.ReadImage(data_directory + "data/deep_learning_contour_mask.nrrd"))
    rs = sitk.GetArrayFromImage(
        sitk.ReadImage(data_directory + "data/deep_learning_contour_mask_reference_standard.nrrd"))


    if mode == "Train":
        cts = np.sum(np.sum(rs, axis=0), 0)
        idx = np.where(cts > 1000)[0]
        np.random.shuffle(idx)

        idx_train = idx[0:800]
        idx_valid = idx[800:1000]
        idx_test = idx[1000:1704]

        np.save(output_directory + "/indexes.npy", [idx_train, idx_valid, idx_test])

        print("Start the training")
        n_t_samples = len(idx_train)
        n_v_samples = len(idx_valid)

        print("Training on %i cells and validating on %i cells" % (n_t_samples, n_v_samples))


        tbCallBack = keras.callbacks.TensorBoard(log_dir=os.path.join(output_directory, 'Graph'),
                                                 histogram_freq=0, write_graph=True, write_images=True)
        saveCallback = keras.callbacks.ModelCheckpoint(os.path.join(output_directory, "unet_model.h5"),
                                                   monitor='val_loss', verbose=0, save_best_only=True,
                                                    save_weights_only=False, mode='auto', period=1)

        print("Creating the model")
        with tf.device('/gpu:%i' % GPU):
            model = get_unet()
            model.fit(x=load_data(imgs, idx_train), y=load_data(rs, idx_train), batch_size=m_batch, epochs=150, verbose=1,
                      callbacks=[tbCallBack, saveCallback],  validation_data=[ load_data(imgs, idx_valid), load_data(rs, idx_valid)],
                      shuffle=True)

            model.save(os.path.join(output_directory,"last_model.h5"))


    if mode == "Test":
        model = get_unet()
        model.load_weights(os.path.join(output_directory, "unet_model.h5"))
        imgsr = np.transpose(imgs, (2, 0, 1))
        imgsr = np.reshape(imgsr, (imgs.shape[2], 200, 200, 1))
        print(imgsr.shape)
        t1 = time.time()
        out = model.predict(imgsr, batch_size=25, verbose=1)
        outr = np.reshape(out, (imgs.shape[2], 200, 200))
        out2 = np.transpose(outr, (1, 2, 0))

        out3 = out2 * 0
        for i in range(0, out2.shape[2]):
            out3[:, :, i] = binary_closing(getLargestCC(out2[:, :, i] > 0.5),
                                           structure=np.zeros((10, 10)) + 1)
        print("Processing time: %f" % (time.time() - t1))

#        sitk.WriteImage(sitk.GetImageFromArray((out3).astype(np.int16)), output_directory + "/predictions.nrrd")


