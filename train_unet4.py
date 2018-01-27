from __future__ import print_function

import os

import numpy as np
from keras import backend as K
from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau
from keras.layers import Input, Conv2D, concatenate, Conv2DTranspose, BatchNormalization, PReLU
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


def dice_coef_sample(y_true, y_pred):
    y_true_f = K.batch_flatten(y_true)
    y_pred_f = K.batch_flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return K.mean((2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth))


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


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def conv_down(ins, n_filters, pooling=True, batch_norm=False):
    conv = Conv2D(n_filters, (3, 3), padding='same', dilation_rate=(2, 2))(ins)
    if batch_norm:
        conv = BatchNormalization(axis=3)(conv)
    conv = PReLU()(conv)
    conv = Conv2D(n_filters, (3, 3), padding='same', dilation_rate=(2, 2))(conv)
    if batch_norm:
        conv = BatchNormalization(axis=3)(conv)
    conv = PReLU()(conv)
    pool = None
    if pooling:
        # pool = MaxPool2D(pool_size=(2, 2))(conv)
        pool = Conv2D(n_filters, (2, 2), strides=(2, 2))(conv)

    return conv, pool


def conv_up(ins, n_filters, skip, batch_norm=False):
    # up = concatenate([UpSampling2D(size=(2, 2))(ins), skip], axis=3)
    conv = Conv2DTranspose(n_filters, (3, 3), padding='same', strides=(2, 2))(ins)
    if batch_norm:
        conv = BatchNormalization(axis=3)(conv)
    conv = PReLU()(conv)
    conv = concatenate([conv, skip], axis=3)
    conv = Conv2D(n_filters, (3, 3), padding='same', dilation_rate=(2, 2))(conv)
    if batch_norm:
        conv = BatchNormalization(axis=3)(conv)
    conv = PReLU()(conv)

    return conv


def get_unet4():
    inputs = Input((img_rows, img_cols, 1))
    n_filters_base = 16
    batch_norm = True
    conv1, pool1 = conv_down(inputs, n_filters_base, pooling=True, batch_norm=batch_norm)
    conv2, pool2 = conv_down(pool1, n_filters_base * 2, pooling=True, batch_norm=batch_norm)
    conv3, pool3 = conv_down(pool2, n_filters_base * 4, pooling=True, batch_norm=batch_norm)
    conv4, pool4 = conv_down(pool3, n_filters_base * 8, pooling=True, batch_norm=batch_norm)

    conv5, _ = conv_down(pool4, n_filters_base * 16, pooling=False)

    conv6 = conv_up(conv5, n_filters_base * 8, conv4, batch_norm=batch_norm)
    conv7 = conv_up(conv6, n_filters_base * 4, conv3, batch_norm=batch_norm)
    conv8 = conv_up(conv7, n_filters_base * 2, conv2, batch_norm=batch_norm)
    conv9 = conv_up(conv8, n_filters_base, conv1, batch_norm=batch_norm)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    model.compile(optimizer=Adam(lr=1e-5, decay=1e-5), loss=dice_coef_loss,
                  metrics=[dice_coef, 'accuracy', precision, recall, f1score])
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
    model = get_unet4()

    csv_logger = CSVLogger('log_unet4_' + str(bit) + '.csv')
    model_checkpoint = ModelCheckpoint('weights_unet4_' + str(bit) + '.h5', monitor='val_loss', save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1)

    print('-' * 30)
    print('Fitting model (bit = ' + str(bit) + ') ...')
    print('-' * 30)

    batch_size = 32

    model.fit(imgs_bit_train, imgs_bit_mask_train, batch_size=batch_size, epochs=epochs, verbose=1, shuffle=True,
              validation_split=0.2,
              callbacks=[csv_logger, model_checkpoint, reduce_lr])

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
    model.load_weights('weights_unet4_' + str(bit) + '.h5')

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
            imsave(os.path.join(pred_dir, str(image_id).split('/')[-1] + '_pred_unet4.png'), image)

    elif bit == 16:
        print('-' * 30)
        print('Saving predicted masks to files...')
        print('-' * 30)
        pred_dir = 'preds_16'
        if not os.path.exists(pred_dir):
            os.mkdir(pred_dir)
        for image, image_id in zip(imgs_mask_test, imgs_bit_id_test):
            image = (image[:, :, 0] * 255.).astype(np.uint8)
            imsave(os.path.join(pred_dir, str(image_id).split('/')[-1] + '_pred_unet4.png'), image)


if __name__ == '__main__':
    train_and_predict(8)
    train_and_predict(16)
