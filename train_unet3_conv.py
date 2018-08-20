from __future__ import print_function

import os

import numpy as np
from keras import backend as K
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.layers import MaxPooling2D, UpSampling2D, Convolution2D, Input, merge, concatenate
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from skimage.io import imsave

from data import load_train_data, load_test_data

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

img_rows = 96
img_cols = 128

smooth = 1.
epochs = 200

def merge(inputs, mode, concat_axis=-1):
    return concatenate(inputs, concat_axis)

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


def get_conv(f=16):
    inputs = Input((img_rows, img_cols, 1))

    conv1 = Convolution2D(f, 3, 3, activation='relu', border_mode='same')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Convolution2D(f, 3, 3, activation='relu', border_mode='same')(conv1)
    conv2 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Convolution2D(2 * f, 3, 3, activation='relu', border_mode='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = Convolution2D(2 * f, 3, 3, activation='relu', border_mode='same')(conv2)
    conv3 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Convolution2D(4 * f, 3, 3, activation='relu', border_mode='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = Convolution2D(4 * f, 3, 3, activation='relu', border_mode='same')(conv3)
    conv4 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Convolution2D(8 * f, 3, 3, activation='relu', border_mode='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = Convolution2D(8 * f, 3, 3, activation='relu', border_mode='same')(conv4)
    conv5 = MaxPooling2D(pool_size=(2, 2))(conv4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Convolution2D(16 * f, 3, 3, activation='relu', border_mode='same')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Convolution2D(16 * f, 3, 3, activation='relu', border_mode='same')(conv5)

    up1 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=3)
    conv6 = BatchNormalization()(up1)
    conv6 = Convolution2D(8 * f, 3, 3, activation='relu', border_mode='same')(conv6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Convolution2D(8 * f, 3, 3, activation='relu', border_mode='same')(conv6)
    up2 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=3)
    conv7 = BatchNormalization()(up2)
    conv7 = Convolution2D(4 * f, 3, 3, activation='relu', border_mode='same')(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Convolution2D(4 * f, 3, 3, activation='relu', border_mode='same')(conv7)
    up3 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=3)
    conv8 = BatchNormalization()(up3)
    conv8 = Convolution2D(2 * f, 3, 3, activation='relu', border_mode='same')(conv8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Convolution2D(2 * f, 3, 3, activation='relu', border_mode='same')(conv8)
    up4 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=3)
    conv9 = BatchNormalization()(up4)
    conv9 = Convolution2D(f, 3, 3, activation='relu', border_mode='same')(conv9)
    conv9 = BatchNormalization()(conv9)
    outputs = Convolution2D(1, 1, 1, activation='hard_sigmoid', border_mode='same')(conv9)
    outputs = Convolution2D(1, 11, 11, activation='hard_sigmoid', border_mode='same')(outputs)

    net = Model(inputs=inputs, outputs=outputs)
    net.compile(loss=dice_coef_loss, optimizer='adam', metrics=[dice_coef, 'accuracy', precision, recall, f1score])

    net.summary()

    return net


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
    model = get_conv(f=16)

    csv_logger = CSVLogger('log_conv_' + str(bit) + '.csv')
    model_checkpoint = ModelCheckpoint('weights_conv_' + str(bit) + '.h5', monitor='val_loss', save_best_only=True)

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
    model.load_weights('weights_conv_' + str(bit) + '.h5')

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
            imsave(os.path.join(pred_dir, str(image_id).split('/')[-1] + '_pred_conv.png'), image)

    elif bit == 16:
        print('-' * 30)
        print('Saving predicted masks to files...')
        print('-' * 30)
        pred_dir = 'preds_16'
        if not os.path.exists(pred_dir):
            os.mkdir(pred_dir)
        for image, image_id in zip(imgs_mask_test, imgs_bit_id_test):
            image = (image[:, :, 0] * 255.).astype(np.uint8)
            imsave(os.path.join(pred_dir, str(image_id).split('/')[-1] + '_pred_conv.png'), image)


if __name__ == '__main__':
    train_and_predict(8)
    train_and_predict(16)
