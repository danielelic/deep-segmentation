from __future__ import print_function

import os

import numpy as np
from skimage.io import imread, imsave
from skimage.transform import resize
from sklearn.cross_validation import train_test_split

data_path = '.'
raw_data_path = os.path.join(data_path, 'raw')
npy_data_path = os.path.join(data_path, 'npy')

image_rows = 96
image_cols = 128


def preprocess(imgs, bit_image=8):
    if bit_image == 8:
        imgs_p = np.ndarray((imgs.shape[0], image_rows, image_cols), dtype=np.uint8)
    else:
        imgs_p = np.ndarray((imgs.shape[0], image_rows, image_cols), dtype=np.uint16)
    for i in range(imgs.shape[0]):
        imgs_p[i] = resize(imgs[i], (image_rows, image_cols), preserve_range=True)
    return imgs_p[..., np.newaxis]


def getData(path, foldlist):
    images = []
    images16 = []
    masks = []
    ids = []
    for i, image_path in enumerate(foldlist):
        image_id = '_'.join(image_path.split('_')[:2])
        image_id = '/'.join(image_id.split('/')[-2:])

        file_name = image_path.split('/')[-1].split('_mask')[0]

        mask = imread(os.path.join(path, file_name + '_mask' + '.png'), as_grey=True)
        masks.append(mask)

        image = imread(os.path.join(path, file_name + '_8' + '.png'), as_grey=True)
        images.append(image)

        image16 = imread(os.path.join(path, file_name + '.png'), as_grey=True)
        images16.append(image16)

        if image_id not in ids:
            ids.append(image_id)

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, len(foldlist)))

    print('Loading done.')

    assert len(images) == len(masks)
    assert len(images) == len(ids), print(len(images), len(ids))
    images = np.array(images, dtype=np.uint8)
    images16 = np.array(images, dtype=np.uint16)
    masks = np.array(masks, dtype=np.uint8)

    masks = preprocess(masks, 8)
    images = preprocess(images, 8)
    images16 = preprocess(images16, 16)

    ids = np.array(ids, dtype=object)

    return images, images16, masks, ids


def create_train_test_data():
    all_images_path = []
    images_dir = 'train'
    for image_name in os.listdir(os.path.join(raw_data_path, images_dir)):
        image_path = os.path.join(raw_data_path, images_dir, image_name)
        all_images_path.append(image_path)

    train_list, test_list = train_test_split([x for x in all_images_path if '_mask' in x], test_size=0.1)

    train_images, train_images16, train_masks, train_ids = getData(os.path.join(raw_data_path, images_dir), train_list)
    test_images, test_images16, test_masks, test_ids = getData(os.path.join(raw_data_path, images_dir), test_list)

    if not os.path.exists(npy_data_path):
        os.mkdir(npy_data_path)

    np.save(os.path.join(npy_data_path, 'images_train.npy'), train_images)
    np.save(os.path.join(npy_data_path, 'images16_train.npy'), train_images16)
    np.save(os.path.join(npy_data_path, 'masks_train.npy'), train_masks)
    np.save(os.path.join(npy_data_path, 'ids_train.npy'), train_ids)

    np.save(os.path.join(npy_data_path, 'images_test.npy'), test_images)
    np.save(os.path.join(npy_data_path, 'images16_test.npy'), test_images16)
    np.save(os.path.join(npy_data_path, 'masks_test.npy'), test_masks)
    np.save(os.path.join(npy_data_path, 'ids_test.npy'), test_ids)
    print('Saving to .npy files done.')


def load_train_data(bit):
    images = np.load(os.path.join(npy_data_path, 'images_train.npy'))
    images16 = np.load(os.path.join(npy_data_path, 'images16_train.npy'))
    masks = np.load(os.path.join(npy_data_path, 'masks_train.npy'))
    ids = np.load(os.path.join(npy_data_path, 'ids_train.npy'))
    if bit == 8:
        return images, masks, ids
    else:
        return images16, masks, ids


def load_test_data(bit):
    images = np.load(os.path.join(npy_data_path, 'images_test.npy'))
    images16 = np.load(os.path.join(npy_data_path, 'images16_test.npy'))
    masks = np.load(os.path.join(npy_data_path, 'masks_test.npy'))
    ids = np.load(os.path.join(npy_data_path, 'ids_test.npy'))
    if bit == 8:
        return images, masks, ids
    else:
        return images16, masks, ids


def dump_predictions(images, ids):
    for image, image_id in zip(images, ids):
        image = (image[:, :, 0] * 255.).astype(np.uint8)
        image = resize(image, (240, 320))
        imsave(os.path.join(raw_data_path, image_id + '_pred.png'), image)


if __name__ == '__main__':
    create_train_test_data()