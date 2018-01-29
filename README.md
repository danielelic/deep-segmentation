# Deep Segmentation

This repository contains several CNNs for semantic segmentation (U-Net, SegNet, ResNet, FractalNet) using Keras library.
The code was developed assuming the use of depth data (e.g. Kinect, Asus Xtion Pro Live).

You can test these scripts on the following datasets:

* [TVHeads (Top-View Heads) Dataset](http://vrai.dii.univpm.it/tvheads-dataset)
* [PIDS (Preterm Infants' Depth Silhouette) Dataset](http://vrai.dii.univpm.it/pids-dataset)

[![YouTubeDemoHeads](https://img.youtube.com/vi/MWjcW-3A5-I/0.jpg)](https://www.youtube.com/watch?v=MWjcW-3A5-I)
[![YouTubeDemoInfant](https://img.youtube.com/vi/_GCnkUXPTJk/0.jpg)](https://www.youtube.com/watch?v=_GCnkUXPTJk)

## Data
Provided data is processed by `data.py` script. This script just loads the images and saves them into NumPy binary format files `.npy` for faster loading later.

```bash
python data.py
```
## Models
The provided models are basically a convolutional auto-encoders.
```
python train_fractal_unet.py
python train_resnet.py
python train_segnet.py
python train_unet.py
python train_unet2.py
python train_unet3_conv.py
python train_unet4.py
```
These deep neural network is implemented with Keras functional API.

Output from the networks is a 96 x 128 which represents mask that should be learned. Sigmoid activation function makes sure that mask pixels are in [0, 1] range.

## Prediction

You can test the online prediction with an OpenNI registration (`.oni` file).
```
python online_prediction.py --v <oni_video_path>
```

### Python Environment Setup

```bash
sudo apt-get install python3-pip python3-dev python-virtualenv # for Python 3.n
virtualenv -p python3 deepseg
. deepseg/bin/activate
```

The preceding command should change your prompt to the following:

```
(deepseg)$ 
```
Install TensorFlow in the active virtualenv environment:

```bash
pip3 install --upgrade tensorflow-gpu # for Python 3.n and GPU
```

Install the others library:

```bash
pip3 install --upgrade keras scikit-learn scikit-image h5py opencv-python primesense
```

### Author
* Daniele Liciotti | [GitHub](https://github.com/danielelic)

### Acknowledgements
* This work is partially inspired by the work of [jocicmarko](https://github.com/jocicmarko).
