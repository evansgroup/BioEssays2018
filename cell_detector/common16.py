import os
import tensorflow as tf
from keras import backend as K
import h5py
import pandas as pd
import numpy as np
import random
import copy

from keras.layers import Input, Conv2D, MaxPooling2D, Activation, BatchNormalization, concatenate, UpSampling2D, Lambda, GlobalAveragePooling2D, Dense
from keras.optimizers import Adam
from keras.models import Model
from keras.optimizers import Adam
import keras as k

def block_to_grid(block, threshold=0.5):
    idxs = np.argwhere(block[:, :, 1] > threshold)
    grid = np.zeros((len(idxs), 4))
    for i, idx in enumerate(idxs):
        grid[i, 0] = idx[1]
        grid[i, 1] = idx[0]
        grid[i, 2] = block[idx[0], idx[1], 2]
        grid[i, 3] = block[idx[0], idx[1], 3]
    return grid

def grid_to_pnps(grid):
    pnps = np.zeros((grid.shape[0], 4))
    pnps[:, 0] = grid[:, 0] * 16 + 8 + 8 * grid[:, 2]
    pnps[:, 1] = grid[:, 1] * 16 + 8 + 8 * grid[:, 3]
    pnps[:, 2] = 0
    pnps[:, 3] = 1
    return pnps


def pnps_to_block(pnps):
    return grid_to_block(pnps_to_grid(pnps))


def block_to_pnps(block, threshold=0.5):
    return grid_to_pnps(block_to_grid(block, threshold))


def pnps_to_grid(pnps, dtype=np.float32):
    grid = np.zeros((pnps.shape[0], 4))
    if len(pnps > 0):
        grid[:, 0] = np.floor(pnps[:, 0] / 16)  # y
        grid[:, 1] = np.floor(pnps[:, 1] / 16)  # x
        grid[:, 2] = (pnps[:, 0] - grid[:, 0] * 16 - 8) / 8  # yd
        grid[:, 3] = (pnps[:, 1] - grid[:, 1] * 16 - 8) / 8  # xd
    return grid


def grid_to_block(grid):
    block = np.zeros((39, 52, 4), dtype=np.float32)
    block[:, :,  0] = 1
    for gf in grid:
        g = gf.astype(np.int16)
        block[g[1], g[0], 0] = 0
        block[g[1], g[0], 1] = 1
        block[g[1], g[0], 2] = gf[2]  # These need to be floats
        block[g[1], g[0], 3] = gf[3]
    return block


# data handler.py
class DataHandler:
    def __init__(self, h5_file, shuffle=False, augmented=False, nb_elements=3,
                 start_idx = 0, end_idx=None,  balance_samples=False):
        self.pnps = pd.read_hdf(h5_file)
        self.data = h5py.File(h5_file, 'r')
        self.imgs = self.data['imgs']
        self.shuffle = shuffle
        self.augmented = augmented
        self.shape = self.imgs.shape
        self.nb_elements = nb_elements
        self.start_idx = start_idx
        if end_idx is None:
            self.end_idx = self.data["imgs"].shape[0]
        else:
            self.end_idx = end_idx
        self.size = self.end_idx - self.start_idx
        self.idxs = self.__idxs_gen()

    def set_nb_elements(self, nb_elements):
        self.nb_elements = nb_elements

    def shape(self):
        return self.shape

    def steps_per_epoch(self):
        return self.size / self.nb_elements + (self.size % self.nb_elements > 0)

    def __idxs_gen(self):
        # Balances the samples in this dataset
        tmp_idxs = np.arange(self.start_idx, self.end_idx, dtype=np.int16)
        if self.shuffle:
            np.random.shuffle(tmp_idxs)
        return list(tmp_idxs)

    # LR data augmentation
    def __augment_lr(self, img, pnps):
        img_lr = img[:, ::-1, :]
        pnps_lr = pnps.copy()
        pnps_lr[:, 0] = img.shape[1] - pnps_lr[:, 0]
        return img_lr, pnps_lr

    # UD data augmentation
    def __augment_ud(self, img, pnps):
        img_ud = img[::-1, :, :]
        pnps_ud = pnps.copy()
        pnps_ud[:, 1] = img.shape[0] - pnps_ud[:, 1]
        return img_ud, pnps_ud

    # LR UD data augmentation
    def __augment_lr_ud(self, img, pnps):
        img_lr_ud = img[::-1, ::-1, :]
        pnps_lr_ud = pnps.copy()
        pnps_lr_ud[:, 1] = img.shape[0] - pnps_lr_ud[:, 1]
        pnps_lr_ud[:, 0] = img.shape[1] - pnps_lr_ud[:, 0]
        return img_lr_ud, pnps_lr_ud

    def __augment_translation(self, img, pnps, dx, dy):
        stack_moved = img.copy()
        if dx >= 0:
            stack_moved[0:img.shape[0] - dx, :, :] = stack_moved[dx::, :, :]
        else:
            stack_moved[-dx::, :, :] = stack_moved[0:(img.shape[0] + dx), :, :]
        if dy >= 0:
            stack_moved[:, 0:img.shape[1] - dy, :] = stack_moved[:, dy::, :]
        else:
            stack_moved[:, -dy::, :] = stack_moved[:, 0:(img.shape[1] + dy), :]

        pnps_moved = pnps.copy()
        pnps_moved[:, 0] = pnps_moved[:, 0] - dy
        pnps_moved[:, 1] = pnps_moved[:, 1] - dx
        pnps_moved_limited = []
        for p in pnps_moved:
            if ((p[0] < 0) or (p[0] >= img.shape[1]) or
                    (p[1] < 0) or (p[1] >= img.shape[0])):
                continue
            pnps_moved_limited.append(p)
        pnps_moved_limited = np.array(pnps_moved_limited)
        return stack_moved, pnps_moved_limited

    def __augment_contrast(self, img, f, a):
        img_ret = img.copy().astype(np.float32) * f + a
        img_ret[img_ret < 0] = 0
        img_ret[img_ret >= 16383] = 16383
        return img_ret

    def augment(self, im, lb):
        mode = random.randint(0, 3)
        if mode == 1:
            im, lb = self.__augment_lr(im, lb)
        if mode == 2:
            im, lb = self.__augment_ud(im, lb)
        if mode == 3:
            im, lb = self.__augment_lr_ud(im, lb)
        dx = random.randint(-35, 35)
        dy = random.randint(-35, 35)
        im, lb = self.__augment_translation(im, lb, dx, dy)
        im = self.__augment_contrast(im, random.uniform(0.75, 1.25), random.randint(-250, 250))
        return im, lb

    def get_sample(self):
        nb_idxs = []
        if self.nb_elements <= len(self.idxs):
            for n in range(self.nb_elements):
                nb_idxs.append(self.idxs.pop(0))
            if len(self.idxs) == 0:
                self.idxs = self.__idxs_gen()
        else:
            nb_idxs = np.array(self.idxs)
            self.idxs = self.__idxs_gen()

        x = np.zeros((len(nb_idxs), 624, 832, 3), dtype=np.float32)
        y = np.zeros((len(nb_idxs), 39,  52 , 4), dtype=np.float32)

        for ix in range(len(nb_idxs)):
            im = self.imgs[nb_idxs[ix]].copy()[:, :, :]*1.0
            lb = copy.deepcopy(self.pnps.iloc[nb_idxs[ix]].pnps)
            if self.augmented:
                im, lb = self.augment(im, lb)
            x[ix, 0:im.shape[0], 0:im.shape[1], :] = (im*1.0) / 5000.0
            y[ix, :, :, :] = grid_to_block(pnps_to_grid(lb))
        return x, y

    def gen_batch(self):
        while 1:
            x, y = self.get_sample()
            yield (x, y)

            

def custom_loss(y_true, y_pred):
    y_true_p1 = K.flatten(y_true[:, :, :, 1])
    y_true_p0 = K.flatten(y_true[:, :, :, 0])
    y_pred_p_w = K.flatten(y_pred[:, :, :, 1]) + K.flatten(y_pred[:, :, :, 0])
    y_pred_p1 = K.flatten(y_pred[:, :, :, 1]) / y_pred_p_w
    y_pred_p0 = K.flatten(y_pred[:, :, :, 0]) / y_pred_p_w
    y_true_p = K.stack([y_true_p0, y_true_p1], axis=1)
    y_pred_p = K.stack([y_pred_p0, y_pred_p1], axis=1)
    l_bce = tf.reduce_mean(k.losses.binary_crossentropy(y_true_p, y_pred_p))

    return l_bce


def custom_loss_l2(y_true, y_pred):
    y_true_p1 = K.flatten(y_true[:, :, :, 1])
    y_true_p0 = K.flatten(y_true[:, :, :, 0])
    y_pred_p_w = K.flatten(y_pred[:, :, :, 1]) + K.flatten(y_pred[:, :, :, 0])
    y_pred_p1 = K.flatten(y_pred[:, :, :, 1]) / y_pred_p_w
    y_pred_p0 = K.flatten(y_pred[:, :, :, 0]) / y_pred_p_w
    y_true_p = K.stack([y_true_p0, y_true_p1], axis=1)
    y_pred_p = K.stack([y_pred_p0, y_pred_p1], axis=1)
    l_bce = tf.reduce_mean(k.losses.binary_crossentropy(y_true_p, y_pred_p))


    y_true_d = K.stack([
        K.flatten(y_true[:, :, :, 2]),
        K.flatten(y_true[:, :, :, 3])], axis=1)
    y_pred_d = K.stack([
        K.flatten(y_pred[:, :, :, 2]),
        K.flatten(y_pred[:, :, :, 3])], axis=1)
    
    l_l2 = tf.sqrt(tf.reduce_sum(k.losses.mean_squared_error(y_true_d, y_pred_d) * y_true_p1) / (
        tf.reduce_sum(y_true_p1) + 1e-7))
    return l_bce + 1.0*l_l2



def sqe(ip):
    x = ip[0]
    y = ip[1]
    e_fc = K.expand_dims(K.expand_dims(y, 1), 1)
    r = K.repeat_elements(e_fc, x.shape[1], axis=1) 
    r = K.repeat_elements(r, x.shape[2], axis=2) 
    return x*r


def get_model_simple():
    inputs = Input((624, 832, 3))
    conv11 = Activation('relu')(BatchNormalization()(Conv2D(16, (5, 5), padding='same')(inputs)))
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv11) #312, 416
    conv21 = Activation('relu')(BatchNormalization()(Conv2D(16, (5, 5), padding='same')(pool1)))
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv21) #156, 208
    conv31 = Activation('relu')(BatchNormalization()(Conv2D(16, (5, 5), padding='same')(pool2)))
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv31)  #78, 104
    conv41 = Activation('relu')(BatchNormalization()(Conv2D(16, (5, 5), padding='same')(pool3)))
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv41)  #39, 52    
    conv51 = Activation('relu')(BatchNormalization()(Conv2D(16, (5, 5), padding='same')(pool4)))
    conv_out = Activation('sigmoid')(BatchNormalization()(Conv2D(4, (3, 3), padding='same')(conv51)))
    model = Model(inputs=[inputs], outputs=[conv_out])
    model.compile(loss=custom_loss_l2, optimizer=Adam(lr=1e-2))
    return model            


