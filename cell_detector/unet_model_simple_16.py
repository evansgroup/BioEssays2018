import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import tensorflow as tf
from keras import backend as K

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
K.set_session(session)

from common16 import pnps_to_grid, grid_to_block, DataHandler, get_model_simple


from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping, TensorBoard, ReduceLROnPlateau

o_dir = './training/unet_model_simple_16_l2_true/'
if not os.path.exists(o_dir):
    os.makedirs(o_dir)
nm_out = o_dir + '/best_model.h5'

nb_epoch = 100

# At the end of the day this is the executable
dh = DataHandler('PNP/deep_learning/data/cell_shooting_new/db_for_training_cells_shuffle_16.h5',
                 augmented=True, shuffle=True,
                 start_idx=0, end_idx=4000, nb_elements=4)

# This is a terrible hack - how can we open the h5 file twice? To be determined
dhv = DataHandler('PNP/deep_learning/data/cell_shooting_new/db_for_validation_cells_shuffle_16.h5',
                  augmented=False, shuffle=False,
                 start_idx=4000, end_idx=5000)

model_checkpoint = ModelCheckpoint(nm_out, monitor='val_loss', verbose=1, save_best_only=True)
model_TensorBoard = TensorBoard(log_dir=o_dir, histogram_freq=0, write_graph=True, write_images=False)
model_EarlyStopping = EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=10, verbose=0, mode='auto')
model_CVSLogger = CSVLogger(o_dir + 'log.csv', separator=',', append=True)

model = get_model_simple()

model.fit_generator(generator=dh.gen_batch(),
                    steps_per_epoch=dh.steps_per_epoch(),
                    validation_data=dhv.gen_batch(),
                    validation_steps=dhv.steps_per_epoch(),
                    epochs=nb_epoch, max_queue_size=10,
                    callbacks=[model_checkpoint, model_TensorBoard, model_EarlyStopping, model_CVSLogger])


