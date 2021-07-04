![](https://github.com/wwwfan628/nn_compression/raw/main/imgs/eth_logo.png) ![](https://github.com/wwwfan628/nn_compression/raw/main/imgs/tiklogo.png)
# Semester Project: Rethinking Information Encoding in Network Compression
![](https://img.shields.io/badge/python-3.8.5-blue.svg)
![](https://img.shields.io/badge/tensorboard-2.5.0-orange.svg)
![](https://img.shields.io/badge/pytorch-1.7.1-green.svg)


---
This repository is code implementation for semester project `Rethinking Information Encoding in Network Compression`.
The final report can be found here: [paper link](https://github.com/wwwfan628/nn_compression/blob/main/Rethinking_Information_Encoding_in_Network_Compression.pdf).

### Results of Index Optimizer
<div align=center>
<table style="width:100%">
  <tr>
    <th>Dataset</th>
    <th>Kaiming Normal</th>
    <th>Kaiming Uniform</th>
    <th>Sparse</th>
  </tr>
  <tr>
    <td>MNIST</td>
    <td>99.49</td>
    <td>99.34</td>
    <td>99.18</td>
  </tr>
  <tr>
    <td>CIFAR10</td>
    <td>90.94</td>
    <td>93.01</td>
    <td>92.55</td>
  </tr>
  <tr>
    <td>ImageNet</td>
    <td>70.93</td>
    <td>/</td>
    <td>/</td>
  </tr>
  </table>
  </div>

### Results of sortGAN
<div align=center>
<table style="width:100%">
  <tr>
    <th>Original Images</th>
    <th>Disordered Pixels</th>
    <th>Recovered Images</th>
  </tr>
  <tr>
    <td><img src="https://github.com/wwwfan628/nn_compression/raw/main/imgs/real.png" width=100% /></td>
    <td><img src="https://github.com/wwwfan628/nn_compression/raw/main/imgs/random.png" width=100% /></td>
    <td><img src="https://github.com/wwwfan628/nn_compression/raw/main/imgs/quantize.png" width=100% /></td>
  </tr>
  </table>
  </div>



## Install

---
To run the code, it's required to install python 3.8.5, tensorboard 2.5.0 and pytorch 1.7.1.
```
$ pip install tensorboard
$ conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
```
ImageNet needs to be downloaded and extracted manually.
```
# download the dataset
$ cd $IMAGENET_DIRECTORY_PATH
$ nohup wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar &
$ nohup wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar &

# extract dataset
$ chmod +x ./src/extract_imagenet.sh
$ nohup ./src/extract_imagenet.sh &

# convert png to jpeg
$ cd $IMAGENET_DIRECTORY_PATH/train
$ mv n02105855/n02105855_2933.JPEG n02105855/n02105855_2933.PNG
$ convert n02105855/n02105855_2933.PNG n02105855/n02105855_2933.JPEG
$ rm n02105855/n02105855_2933.PNG
```

## Usage

---
All of the following commands should be executed within the `./src` subfolder. 
### 1) Preliminary Experiment
Execute the preliminary experiment with the following command:
```
$ python preliminary_experiment.py --dataset_name=MNIST --model_name=LeNet5
```
`--dataset_name` can be either `MNIST` or `CIFAR10`, `--model_name` represents the architecture used in the 
experiment and can be chosen from `LeNet5` or `VGG`.

To plot the weight distributions, head over to `notebooks/plot_weight_distribution.ipynb` or directly check the 
tensorboard log with following command:
```
$ tensorboard --logdir=runs/$LOG_DIRECTORY_NAME
```
### 2) Index Optimizer
Experiments, where neural networks are initialized with Kaiming normal or Kaiming uniform 
distributions, are implemented in script `train_experiment.py`. 
```
$ python train_experiment.py --dataset_name=MNIST --model_name=LeNet5 --train_index 
                             [--ste] [--normal] [--granularity_channel] [--granularity_kernel]
                             [--max_epoch=100] [--patience=20]
```
You can turn on/off the optional functions of index optimizer by adding the corresponding optional arguments.

Experiments, where neural networks are pruned before training, are implementd in 
script `prune_experiment.py`. The optional arguments can be added similarly as above.
```
$ python prune_experiment.py --dataset_name=MNIST --model_name=LeNet5 --train_index 
                             [--ste] [--normal] [--granularity_channel] [--granularity_kernel]
                             [--max_epoch=100] [--patience=20]
```

### 3) Input Permutation Experiments

* Train LeNet5 on MNIST with command in `experiment 2)`. Store the final parameters under `$CHECKPOINT_PATH1`.
Repeat this step two more times.

  
* Train the prototype of discriminator with following command:
```
$ python train_discrminator.py --dataset_name=MNIST --model_name=LeNet5 
                             [--train_index] [--ste] [--max_epoch] [--patience]
```
Store the final parameter under `$DISCRIMINATOR_CHECKPOINT_PATH1`. Repeat this step two more times.


* Run input permutation experiment using:
```
$ python input_experiment.py --dataset_name=MNIST --image_index=0 
                             --checkpoint_path_1=$CHECKPOINT_PATH1
                             --checkpoint_path_2=$CHECKPOINT_PATH2
                             --checkpoint_path_3=$CHECKPOINT_PATH3
                             --discriminator_checkpoint_path_1=$DISCRIMINATOR_CHECKPOINT_PATH1
                             --discriminator_checkpoint_path_2=$DISCRIMINATOR_CHECKPOINT_PATH2
                             --discriminator_checkpoint_path_3=$DISCRIMINATOR_CHECKPOINT_PATH3
                             [--lr] [--max_epoch] [--patience]
```
`--image_index` is the index of the image sample that you want to recover. Provide the paths, where you store the final 
parameters of LeNet5, after `--checkpoint_path_n`. Similarly, provide the paths that store parameters of discriminator 
after `--discriminator_checkpoint_path_n`.

### 4) sortGAN
Use following command to run experiment of sortGAN. `--dataset_name` can be set to `MNIST` or `FashionMNIST`.
```
$ python input_experiment_cGAN.py --dataset_name=MNIST [--max_epoch] [--lr]
```

## Credits

---
This implementation was developed by [Yifan Lu](https://github.com/wwwfan628). Please contact Yifan for any enquiry.
