from __future__ import print_function

import os

import numpy as np
from keras import backend as K
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.layers import Convolution2D, UpSampling2D, AveragePooling2D, SpatialDropout2D, merge, Input, concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras.optimizers import Adam
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

def get_unet2():
    input = Input((img_rows, img_cols, 1))
    conv1 = Convolution2D(32, 3, 3, border_mode='same', init='he_normal')(input)
    conv1 = LeakyReLU()(conv1)
    conv1 = SpatialDropout2D(0.2)(conv1)
    conv1 = Convolution2D(32, 3, 3, border_mode='same', init='he_normal')(conv1)
    conv1 = LeakyReLU()(conv1)
    conv1 = SpatialDropout2D(0.2)(conv1)
    pool1 = AveragePooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(64, 3, 3, border_mode='same', init='he_normal')(pool1)
    conv2 = LeakyReLU()(conv2)
    conv2 = SpatialDropout2D(0.2)(conv2)
    conv2 = Convolution2D(64, 3, 3, border_mode='same', init='he_normal')(conv2)
    conv2 = LeakyReLU()(conv2)
    conv2 = SpatialDropout2D(0.2)(conv2)
    pool2 = AveragePooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(128, 3, 3, border_mode='same', init='he_normal')(pool2)
    conv3 = LeakyReLU()(conv3)
    conv3 = SpatialDropout2D(0.2)(conv3)
    conv3 = Convolution2D(128, 3, 3, border_mode='same', init='he_normal')(conv3)
    conv3 = LeakyReLU()(conv3)
    conv3 = SpatialDropout2D(0.2)(conv3)

    comb1 = merge([conv2, UpSampling2D(size=(2, 2))(conv3)], mode='concat', concat_axis=3)
    conv4 = Convolution2D(64, 3, 3, border_mode='same', init='he_normal')(comb1)
    conv4 = LeakyReLU()(conv4)
    conv4 = SpatialDropout2D(0.2)(conv4)
    conv4 = Convolution2D(64, 3, 3, border_mode='same', init='he_normal')(conv4)
    conv4 = LeakyReLU()(conv4)
    conv4 = SpatialDropout2D(0.2)(conv4)

    comb2 = merge([conv1, UpSampling2D(size=(2, 2))(conv4)], mode='concat', concat_axis=3)
    conv5 = Convolution2D(32, 3, 3, border_mode='same', init='he_normal')(comb2)
    conv5 = LeakyReLU()(conv5)
    conv5 = SpatialDropout2D(0.2)(conv5)
    conv5 = Convolution2D(32, 3, 3, border_mode='same', init='he_normal')(conv5)
    conv5 = LeakyReLU()(conv5)
    conv5 = SpatialDropout2D(0.2)(conv5)

    output = Convolution2D(1, 1, 1, activation='sigmoid')(conv5)

    model = Model(input=input, output=output)
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
    model = get_unet2()

    csv_logger = CSVLogger('log_unet2_' + str(bit) + '.csv')
    model_checkpoint = ModelCheckpoint('weights_unet2_' + str(bit) + '.h5', monitor='val_loss', save_best_only=True)

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
    model.load_weights('weights_unet2_' + str(bit) + '.h5')

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
            imsave(os.path.join(pred_dir, str(image_id).split('/')[-1] + '_pred_unet2.png'), image)

    elif bit == 16:
        print('-' * 30)
        print('Saving predicted masks to files...')
        print('-' * 30)
        pred_dir = 'preds_16'
        if not os.path.exists(pred_dir):
            os.mkdir(pred_dir)
        for image, image_id in zip(imgs_mask_test, imgs_bit_id_test):
            image = (image[:, :, 0] * 255.).astype(np.uint8)
            imsave(os.path.join(pred_dir, str(image_id).split('/')[-1] + '_pred_unet2.png'), image)


if __name__ == '__main__':
    train_and_predict(8)
    train_and_predict(16)
