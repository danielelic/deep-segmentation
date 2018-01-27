# deep-segmentation

This repository contains several CNNs for semantic segmentation using Keras library.

# Python Environment Setup

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

## Author
* Daniele Liciotti | [GitHub](https://github.com/danielelic)
