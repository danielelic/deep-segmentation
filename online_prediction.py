import argparse

import numpy as np
from PIL import Image
from primesense import openni2
from skimage.transform import resize

from train_unet3_conv import get_conv

img_rows = 96
img_cols = 128

if __name__ == '__main__':
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description="")
    p.add_argument('--v', dest='video_path', action='store', default='', help='path Video')
    args = p.parse_args()

    model = get_conv()
    bit = 16

    model.load_weights('weights_conv_' + str(bit) + '2.h5')

    dev = openni2.Device
    try:
        openni2.initialize()
        dev = openni2.Device.open_file(args.video_path)
        print(dev.get_sensor_info(openni2.SENSOR_DEPTH))
    except (RuntimeError, TypeError, NameError):
        print(RuntimeError, TypeError, NameError)

    pbs = openni2.PlaybackSupport(dev)
    depth_stream = pbs.device.create_depth_stream()

    pbs.set_repeat_enabled(True)
    pbs.set_speed(-1.0)
    depth_stream.start()

    n_frames = pbs.get_number_of_frames(depth_stream)
    for i in range(0, n_frames - 1):
        frame_depth = depth_stream.read_frame()
        print("Depth {0} of {1} - {2}".format(i, n_frames, frame_depth.frameIndex))
        frame_depth_data = frame_depth.get_buffer_as_uint16()
        depth_array = np.ndarray((frame_depth.height, frame_depth.width),
                                 dtype=np.uint16,
                                 buffer=frame_depth_data)
        depth_array = resize(depth_array, (img_rows, img_cols), preserve_range=True)
        imgs = np.array([depth_array], dtype=np.uint16)
        imgs = imgs[..., np.newaxis]

        imgs = imgs.astype('float32')

        mean = np.mean(imgs)
        std = np.std(imgs)

        imgs -= mean
        imgs /= std

        np.ndarray((imgs.shape[0], img_rows, img_cols), dtype=np.uint16)
        predicted_image = model.predict(imgs, verbose=0)
        image = (predicted_image[0][:, :, 0] * 255.).astype(np.uint8)
        img = Image.fromarray(image)
        img.save("./predicted_images/" + str(i).zfill(4) + ".png")

    depth_stream.stop()
    openni2.unload()
