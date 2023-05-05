# Deep-Object-Pose (DOPE)

This is an algorithm repo and format to ros package let everyone can use it easily, if you want to use, please submodule this repo and create a new branch for your project.

## Docker images for DOPE

### Build docker images

#### For GPU
```
    $ cd Deep-Object-Pose/Docker/GPU && source build.sh
```

#### For TX2
```
    $ cd Deep-Object-Pose/Docker/TX2 && source build.sh
```

### Pull docker images
```
    $ docker pull yimlaisum2014/dope:gpu-noetic
```

## Usage
### 1. Enter docker
```
    $ source docker_run.sh
```

### 2. Dowload dataaset
Dowload sample dataset from [NAS](http://gofile.me/773h8/Uxcbszrg1), which generated from this [repo](https://github.com/ARG-NCTU/robotx2022-unity-dataset) and unzip it and put it under Deep-Object-Pose/dataset/.

### 3. Train
```
    $ python3 scripts/train.py --data /home/arg/Deep-Object-Pose/dataset/LIVALO_train/ --workers 1 --batchsize 5 --namefile LIVALO --gpuids 0 --outf LIVALO --epochs 10
```

## Simple example for training and inference about DOPE

The training code 
\
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1wJpa9XDxWlob1hUvKKXEXLcUf98nVs7X#scrollTo=ibQa3N_bgYON)

The inference code
\
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1dRdNBlwD5XScuRQpakYEfFoORJU62LYs)
