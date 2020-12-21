from modelGenerator import build_model
from keras.callbacks import TensorBoard, ModelCheckpoint
from cfg import IMSIZE, MNAME, DATA_DIR, epoch_num,  batchSize
from dataGenerator import generate_data, train_data_generator,val_data_generator
import tensorflow as tf
import time
import os
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = '0'   #指定第一块GPU可用
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9  # 程序最多只能占用指定gpu50%的显存
config.gpu_options.allow_growth = True      #程序按需申请内存
sess = tf.compat.v1.Session(config = config)

tbd = TensorBoard(log_dir=f'./logs/MRI-{time.time()}')
mckpt = ModelCheckpoint(
                filepath="./ckpt/MRI-3D.hdf5",
                save_best_only=True,  # Only save a model if `loss` has improved.
                monitor="accuracy",
                verbose=1,
            )
callbacks = [tbd, mckpt]


model = build_model(22)
model._get_distribution_strategy = lambda: None
history = model.fit_generator(train_data_generator(DATA_DIR, IMSIZE, batchSize),steps_per_epoch = np.floor(len(generate_data(DATA_DIR, IMSIZE)[0])/batchSize), epochs=epoch_num,  callbacks=callbacks)
model.save('./model/MRI-3D')

x_val, y_val =val_data_generator(DATA_DIR, IMSIZE)
loss, acc=model.evaluate(x_val,y_val)
print(f'loss:{loss},accuracy:{acc}')