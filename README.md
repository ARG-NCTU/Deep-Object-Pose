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
### Train 

1. Dowload dataaset
Dowload sample dataset from [NAS](http://gofile.me/773h8/Uxcbszrg1), which generated from this [repo](https://github.com/ARG-NCTU/robotx2022-unity-dataset) and unzip it and put it under Deep-Object-Pose/dataset/.

2. Enter docker
```
    $ source docker_run.sh
```
3. Dowload Dataset
Dwnload dataset with **download_dataset.sh**  and unzip it, put it under Deep-Object-Pose/dataset/.   
Or you can use this [repo](https://github.com/ARG-NCTU/robotx2022-unity-dataset) to generate your own dataset.
```
   $ source download_dataset.sh 
```
<!-- Dowload sample dataset from [NAS](http://gofile.me/773h8/Uxcbszrg1), which generated from this [repo](https://github.com/ARG-NCTU/robotx2022-unity-dataset) and unzip it and put it under Deep-Object-Pose/dataset/. -->

4. Train
```
$ python3 scripts/train_with_config.py 
```

## Simple example for training and inference about DOPE
<!-- 3. Train
```
    $ python3 scripts/train.py --data /home/arg-dope/Deep-Object-Pose/dataset/LIVALO_train/ --workers 1 --batchsize 5 --namefile LIVALO --gpuids 0 --outf LIVALO --epochs 10
``` -->
### Test
1. Create catkin_ws and Clone this repo
```
    $ mkdir -p ~/catkin_ws/src
    $ cd ~/catkin_ws/src
    $ git clone https://github.com/yimlaisum2014/Deep-Object-Pose.git dope
    $ cd dope/
```
2. Dowload models and testing data datset 

- Download model to inference about DOPE with **download_model.sh** , and put it under catkin_ws/src/dope/weights/ .
```
    $ source download_model.sh
```
3. Enter Docker and make&source workspace
```
    $ source docker_run_4_inference.sh
    $ cd /home/catkin_ws/
    $ catkin_make
    $ source devel/setup.bash
```
4. roslaunch dope dope.launch
- Give weights and other info in **config/config_pose_robotx.yaml**  
- Also decide InputPath, OutputPath in **nodes/config/path.yaml**  
```
    $ roslaunch dope dope_local.launch
```

## Simple example for training and inference about DOPE

The training code 
\
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1wJpa9XDxWlob1hUvKKXEXLcUf98nVs7X#scrollTo=ibQa3N_bgYON)

The inference code
\
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1dRdNBlwD5XScuRQpakYEfFoORJU62LYs)
