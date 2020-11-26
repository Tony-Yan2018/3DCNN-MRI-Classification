import keras.backend as K
import tensorflow as tf
from keras.layers import Input, Conv2D, BatchNormalization, UpSampling2D, Flatten, Dense, MaxPool3D, Conv3D, GlobalAveragePooling3D, Dropout
from keras.layers.merge import concatenate
from keras.models import Model
from keras.utils import plot_model
from keras.losses import categorical_crossentropy
from cfg import IMSIZE, MNAME



def build_model(depth):
    input = Input(shape=(IMSIZE, IMSIZE, depth, 1))
    input = Input(shape=(IMSIZE, IMSIZE, depth, 1))

    layer = Conv3D(filters=64, kernel_size=3, name='STIR_3d_1', activation="relu")(input)
    layer = Conv3D(filters=64, kernel_size=3, name='STIR_3d_2', activation="relu")(layer)
    layer = BatchNormalization()(layer)

    layer = Conv3D(filters=128, kernel_size=3, name='STIR_3d_3', activation="relu")(layer)
    layer = MaxPool3D(pool_size=2)(layer)
    layer = BatchNormalization()(layer)

    layer = Conv3D(filters=256, kernel_size=3, name='STIR_3d_4', activation="relu")(layer)
    layer = MaxPool3D(pool_size=2)(layer)
    layer = BatchNormalization()(layer)

    layer = GlobalAveragePooling3D()(layer)
    layer = Dense(units=512, name='STIR_dense',activation="relu")(layer)
    layer = Dropout(0.3)(layer)

    # T1 = Conv3D(filters=64, kernel_size=3, name='T1_3d_1',activation="relu")(inputT1)
    # # T1 = MaxPool3D(pool_size=2)(T1)
    # # T1 = BatchNormalization()(T1)
    #
    # T1 = Conv3D(filters=64, kernel_size=3, name='T1_3d_2', activation="relu")(T1)
    # # T1 = MaxPool3D(pool_size=2)(T1)
    # T1 = BatchNormalization()(T1)
    #
    # T1 = Conv3D(filters=128, kernel_size=3, name='T1_3d_3', activation="relu")(T1)
    # T1 = MaxPool3D(pool_size=2)(T1)
    # T1 = BatchNormalization()(T1)
    #
    # T1 = Conv3D(filters=256, kernel_size=3,  name='T1_3d_4',activation="relu")(T1)
    # # print(T1.shape)
    # T1 = MaxPool3D(pool_size=2)(T1)
    # T1 = BatchNormalization()(T1)
    #
    # T1 = GlobalAveragePooling3D()(T1)
    # T1 = Dense(units=512, name='T1_dense', activation="relu")(T1)
    # T1 = Dropout(0.3)(T1)
    #
    # # STIR = Dense(units=1, activation="sigmoid")(STIR)
    #
    #
    # fuse = concatenate([STIR, T1], axis=2)
    # fuse = Dense(512, name='fusion_1', activation='relu')(fuse)
    # fuse = Dropout(0.4)(fuse)
    layer = Dense(3, name='classification', activation='softmax')(layer)

    model = Model(inputs=input, outputs=layer, name=MNAME)
    model.compile(optimizer='adam', loss=categorical_crossentropy, metrics=['accuracy'])

    print(model.metrics_names)
    print(model.summary())
    plot_model(model, to_file=f'Model_Summary_{MNAME}.png', show_shapes=True)
    return model