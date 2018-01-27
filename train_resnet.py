from __future__ import print_function

import os
import sys

sys.setrecursionlimit(10000)

import numpy as np
from keras import layers
from keras import backend as K
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.layers import Activation
from keras.layers import Input, Conv2D, ZeroPadding2D, MaxPooling2D, UpSampling2D, concatenate
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from skimage.io import imsave

from data import load_train_data, load_test_data

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

img_rows = 96
img_cols = 128

smooth = 1.
epochs = 200


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def f1score(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall))

def identity_block(input_tensor, kernel_size, filters, stage, block):
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides, name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides, name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


def up_conv_block(input_tensor, kernel_size, filters, stage, block, strides=(1, 1)):
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    up_conv_name_base = 'up' + str(stage) + block + '_branch'
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = UpSampling2D(size=(2, 2), name=up_conv_name_base + '2a')(input_tensor)

    x = Conv2D(filters1, (1, 1), strides=strides, name=conv_name_base + '2a')(x)

    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = UpSampling2D(size=(2, 2), name=up_conv_name_base + '1')(input_tensor)
    shortcut = Conv2D(filters3, (1, 1), strides=strides, name=conv_name_base + '1')(shortcut)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


# https://arxiv.org/pdf/1512.03385v1.pdf
# 50 Layer resnet

def get_resnet(f=16, bn_axis=3, in_channels=1, classes=1):
    input = Input((img_rows, img_cols, 1))
    x = ZeroPadding2D((4, 4))(input)
    x = Conv2D(f, (7, 7), strides=(2, 2), name='conv1')(x)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [f, f, f * 2], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [f, f, f * 2], stage=2, block='b')
    x2 = identity_block(x, 3, [f, f, f * 2], stage=2, block='c')

    x = conv_block(x2, 3, [f * 2, f * 2, f * 4], stage=3, block='a')
    x = identity_block(x, 3, [f * 2, f * 2, f * 4], stage=3, block='b')
    # x = identity_block(x, 3, [128, 128, 256], stage=3, block='c')
    x3 = identity_block(x, 3, [f * 2, f * 2, f * 4], stage=3, block='d')

    x = conv_block(x3, 3, [f * 4, f * 4, f * 8], stage=4, block='a')
    x = identity_block(x, 3, [f * 4, f * 4, f * 8], stage=4, block='b')
    # x = identity_block(x, 3, [256, 256, 512], stage=4, block='c')
    # x = identity_block(x, 3, [256, 256, 512], stage=4, block='d')
    # x = identity_block(x, 3, [256, 256, 512], stage=4, block='e')
    x4 = identity_block(x, 3, [f * 4, f * 4, f * 8], stage=4, block='f')

    x = conv_block(x4, 3, [f * 8, f * 8, f * 16], stage=5, block='a')
    x = identity_block(x, 3, [f * 8, f * 8, f * 16], stage=5, block='b')
    x = identity_block(x, 3, [f * 8, f * 8, f * 16], stage=5, block='c')

    x = up_conv_block(x, 3, [f * 16, f * 8, f * 8], stage=6, block='a')
    x = identity_block(x, 3, [f * 16, f * 8, f * 8], stage=6, block='b')
    x = identity_block(x, 3, [f * 16, f * 8, f * 8], stage=6, block='c')

    x = concatenate([x, x4], axis=bn_axis)

    x = up_conv_block(x, 3, [f * 16, f * 4, f * 4], stage=7, block='a')
    x = identity_block(x, 3, [f * 16, f * 4, f * 4], stage=7, block='b')
    # x = identity_block(x, 3, [1024, 256, 256], stage=7, block='c')
    # x = identity_block(x, 3, [1024, 256, 256], stage=7, block='d')
    # x = identity_block(x, 3, [1024, 256, 256], stage=7, block='e')
    x = identity_block(x, 3, [f * 16, f * 4, f * 4], stage=7, block='f')

    x = concatenate([x, x3], axis=bn_axis)

    x = up_conv_block(x, 3, [f * 8, f * 2, f * 2], stage=8, block='a')
    x = identity_block(x, 3, [f * 8, f * 2, f * 2], stage=8, block='b')
    # x = identity_block(x, 3, [512, 128, 128], stage=8, block='c')
    x = identity_block(x, 3, [f * 8, f * 2, f * 2], stage=8, block='d')

    x = concatenate([x, x2], axis=bn_axis)

    x = up_conv_block(x, 3, [f * 4, f, f], stage=10, block='a', strides=(1, 1))
    x = identity_block(x, 3, [f * 4, f, f], stage=10, block='b')
    x = identity_block(x, 3, [f * 4, f, f], stage=10, block='c')

    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(classes, (3, 3), padding='same', activation='sigmoid', name='convLast')(x)

    model = Model(input, x, name='resnetUnet')
    model.compile(optimizer=Adam(lr=3e-4), loss=dice_coef_loss,
                  metrics=[dice_coef, 'accuracy', precision, recall, f1score])

    model.summary()

    return model


def train_and_predict(bit):
    print('-' * 30)
    print('Loading and train data (bit = ' + str(bit) + ') ...')
    print('-' * 30)
    imgs_bit_train, imgs_bit_mask_train, _ = load_train_data(bit)

    print(imgs_bit_train.shape[0], imgs_bit_mask_train.shape[0])

    imgs_bit_train = imgs_bit_train.astype('float32')
    mean = np.mean(imgs_bit_train)
    std = np.std(imgs_bit_train)

    imgs_bit_train -= mean
    imgs_bit_train /= std

    imgs_bit_mask_train = imgs_bit_mask_train.astype('float32')
    imgs_bit_mask_train /= 255.  # scale masks to [0, 1]

    print('-' * 30)
    print('Creating and compiling model (bit = ' + str(bit) + ') ...')
    print('-' * 30)
    model = get_resnet()

    csv_logger = CSVLogger('log_resnet_' + str(bit) + '.csv')
    model_checkpoint = ModelCheckpoint('weights_resnet_' + str(bit) + '.h5', monitor='val_loss', save_best_only=True)

    print('-' * 30)
    print('Fitting model (bit = ' + str(bit) + ') ...')
    print('-' * 30)

    model.fit(imgs_bit_train, imgs_bit_mask_train, batch_size=32, epochs=epochs, verbose=1, shuffle=True,
              validation_split=0.2,
              callbacks=[csv_logger, model_checkpoint])

    print('-' * 30)
    print('Loading and preprocessing test data (bit = ' + str(bit) + ') ...')
    print('-' * 30)

    imgs_bit_test, imgs_mask_test, imgs_bit_id_test = load_test_data(bit)

    imgs_bit_test = imgs_bit_test.astype('float32')
    imgs_bit_test -= mean
    imgs_bit_test /= std

    print('-' * 30)
    print('Loading saved weights...')
    print('-' * 30)
    model.load_weights('weights_resnet_' + str(bit) + '.h5')

    print('-' * 30)
    print('Predicting masks on test data (bit = ' + str(bit) + ') ...')
    print('-' * 30)
    imgs_mask_test = model.predict(imgs_bit_test, verbose=1)

    if bit == 8:
        print('-' * 30)
        print('Saving predicted masks to files...')
        print('-' * 30)
        pred_dir = 'preds_8'
        if not os.path.exists(pred_dir):
            os.mkdir(pred_dir)
        for image, image_id in zip(imgs_mask_test, imgs_bit_id_test):
            image = (image[:, :, 0] * 255.).astype(np.uint8)
            imsave(os.path.join(pred_dir, str(image_id).split('/')[-1] + '_pred_resnet.png'), image)

    elif bit == 16:
        print('-' * 30)
        print('Saving predicted masks to files...')
        print('-' * 30)
        pred_dir = 'preds_16'
        if not os.path.exists(pred_dir):
            os.mkdir(pred_dir)
        for image, image_id in zip(imgs_mask_test, imgs_bit_id_test):
            image = (image[:, :, 0] * 255.).astype(np.uint8)
            imsave(os.path.join(pred_dir, str(image_id).split('/')[-1] + '_pred_resnet.png'), image)


if __name__ == '__main__':
    train_and_predict(8)
    train_and_predict(16)
