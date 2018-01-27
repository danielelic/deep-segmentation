from __future__ import print_function

import os

import numpy as np
from keras import backend as K
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D
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


def get_unet():
    inputs = Input((img_rows, img_cols, 1))
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
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

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
    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss,
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
    model = get_unet()

    csv_logger = CSVLogger('log_unet_' + str(bit) + '.csv')
    model_checkpoint = ModelCheckpoint('weights_unet_' + str(bit) + '.h5', monitor='val_loss', save_best_only=True)

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
    model.load_weights('weights_unet_' + str(bit) + '.h5')

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
            imsave(os.path.join(pred_dir, str(image_id).split('/')[-1] + '_pred.png'), image)

    elif bit == 16:
        print('-' * 30)
        print('Saving predicted masks to files...')
        print('-' * 30)
        pred_dir = 'preds_16'
        if not os.path.exists(pred_dir):
            os.mkdir(pred_dir)
        for image, image_id in zip(imgs_mask_test, imgs_bit_id_test):
            image = (image[:, :, 0] * 255.).astype(np.uint8)
            imsave(os.path.join(pred_dir, str(image_id).split('/')[-1] + '_pred.png'), image)


if __name__ == '__main__':
    train_and_predict(8)
    train_and_predict(16)
