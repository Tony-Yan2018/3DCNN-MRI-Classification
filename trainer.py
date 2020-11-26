from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from modelGenerator import build_model
from keras.callbacks import TensorBoard, ModelCheckpoint
from cfg import IMSIZE, MNAME, DATA_DIR, epoch_num, steps_per_epoch
from dataGenerator import generate_data
import random

tbd = TensorBoard(log_dir=r'C:\logs\MRI-VGG16')
mckpt = ModelCheckpoint(
                filepath="./ckpt/MRI-VGG16.hdf5",
                save_best_only=True,  # Only save a model if `loss` has improved.
                monitor="loss",
                verbose=1,
            )
callbacks = [tbd, mckpt]


model = build_model(22)
model._get_distribution_strategy = lambda: None
history = model.fit_generator(generate_data(DATA_DIR, IMSIZE), epochs=epoch_num, steps_per_epoch=steps_per_epoch, callbacks=callbacks)
model.save('./model/MRI-VGG16')