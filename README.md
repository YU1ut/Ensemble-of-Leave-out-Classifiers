# Ensemble of Self Supervised Leave-out Classifiers
This is an unofficial pytorch implementation of [Out-of-Distribution Detection Using an Ensemble of Self Supervised Leave-out Classifiers](http://openaccess.thecvf.com/content_ECCV_2018/papers/Apoorv_Vyas_Out-of-Distribution_Detection_Using_ECCV_2018_paper.pdf). 

## Requirements
- Python 3.7+
- PyTorch 0.4.1
- torchvision 0.2.0 (**Do not use the lastest 0.2.2 version**)
- progress
- matplotlib
- numpy

## Preparation
Download five out-of-distributin datasets provided by [ODIN](https://github.com/ShiyuLiang/odin-pytorch):

* **[Tiny-ImageNet (crop)](https://www.dropbox.com/s/avgm2u562itwpkl/Imagenet.tar.gz)**
* **[Tiny-ImageNet (resize)](https://www.dropbox.com/s/kp3my3412u5k9rl/Imagenet_resize.tar.gz)**
* **[LSUN (crop)](https://www.dropbox.com/s/fhtsw1m3qxlwj6h/LSUN.tar.gz)** 
* **[LSUN (resize)](https://www.dropbox.com/s/moqh2wh8696c3yl/LSUN_resize.tar.gz)** 
* **[iSUN](https://www.dropbox.com/s/ssz7qxfqae0cca5/iSUN.tar.gz)**

Here is an example code of downloading Tiny-ImageNet (crop) dataset. In the **root** directory, run

```
mkdir data
cd data
wget https://www.dropbox.com/s/avgm2u562itwpkl/Imagenet.tar.gz
tar -xvzf Imagenet.tar.gz
cd ..
```

## Usage

### Pre-trained Models
We provide download links of four types of pre-trained models.
* **[DenseNet-BC trained on CIFAR-10](https://www.dropbox.com/s/mo2xay9rpfk8emu/cifar10_dense.tar.gz)** 
* **[DenseNet-BC trained on CIFAR-100](https://www.dropbox.com/s/oo8fyxlosqkiov6/cifar100_dense.tar.gz)**
* **[Wide ResNet trained on CIFAR-10](https://www.dropbox.com/s/6set3ilyto2adol/cifar10_wide.tar.gz)**
* **[Wide ResNet trained on CIFAR-100](https://www.dropbox.com/s/ig5uftywotlt9i5/cifar100_wide.tar.gz)** 

Each type of models contains 5 models for 5-fold.
Here is an example code of downloading DenseNet-BC trained on CIFAR-10. In the **root** directory, run

```
mkdir checkpoints
cd checkpoints
wget https://www.dropbox.com/s/mo2xay9rpfk8emu/cifar10_dense.tar.gz
tar -xvzf cifar10_dense.tar.gz
cd ..
```

### Train single model (Optional)
Train DenseNet on the first fold of CIFAR-10.
```
python train.py -c checkpoints/cifar10_fold_1_dense_checkpoint --fold 1
```
Trained model will be saved at `checkpoints/cifar10_fold_1_dense_checkpoint`.

### Train all models (Optional)
```
python train_all.py
```
This script will train models of DenseNet/WideResNet on 5-fold CIFAR-10/100 which results in 20 models.
Trained model will be saved at `checkpoints`.

### Test
Use 5-fold ensemble and [ODIN](https://github.com/ShiyuLiang/odin-pytorch) to detect OOD samples.
For example, to test DenseNet-BC trained on CIFAR-10 where TinyImageNet (crop) is the out-of-distribution dataset, please run 
```
python test.py --in-dataset cifar10 --out-dataset Imagenet --magnitud 0.002
```
The temperature is set as 1000, and perturbation magnitude is set as 0.002.

**Note:** Please choose arguments according to the following. 

#### args
* **args.in_dataset**: the arguments of in-distribution datasets are shown as follows
	
	|In-Distribution Datasets | args.in_dataset
	|----------------------|--------
	|CIFAR-10 | cifar10
	|CIFAR-100 | cifar100

* **args.out_dataset**: the arguments of out-of-distribution datasets are shown as follows

	|Out-of-Distribution Datasets     | args.out_dataset
	|------------------------------------|-----------------
	|Tiny-ImageNet (crop)                | Imagenet
	|Tiny-ImageNet (resize)              | Imagenet_resize
	|LSUN (crop)                         | LSUN
	|LSUN (resize)                       | LSUN_resize
	|iSUN                                | iSUN

* **args.magnitude**: the noise magnitude can be found below. You can use these values to reproduce the retults. Please notice that these values are not optimal .

	|Out-of-Distribution Datasets        |   DenseNet on CIFAR-10     |  DenseNet on CIFAR-100  | WideResNet on CIFAR-10   | WideResNet on CIFAR-100
	|------------------------------------|------------------|-------------  | -------------- |--------------
	|Tiny-ImageNet (crop)  | 0.002  | 0.002  | 0.003  | 0.002
	|Tiny-ImageNet (resize)  | 0.002  | 0.002  | 0.003   | 0.002
	|LSUN (crop)  | 0.002 | 0.003 | 0.001 | 0.002
	|LSUN (resize)  | 0.002  | 0.003   | 0.002  | 0.002

* **args.wide**: the arguments of network choices are shown as follows

	|Nerual Network Models     | args.wide
	|------------------------------------|-----------------
	|DenseNet                | False
	|WideResNet | True

* **args.temperature**: temperature is set to 1000 in all cases.

## Results
All values are percentages. ↑ indicates larger value is better, and ↓ indicates lower value is better. Each value cell is in "Paper/Our Implementation" format.
#### DenseNet on CIFAR-10
| OOD Dataset | FPR at 95% TPR ↓ | Detection Error ↓ | AUROC ↑ | AUPR In ↑ | AUPR Out ↑ |
|:---|:---:|:---:|:---:|:---:|:---:|
|Tiny-ImageNet (crop) |1.23 / 1.48|2.63 / 2.79|99.65 / 99.66| 99.68 / 99.67|99.64 / 99.66|
|Tiny-ImageNet (resize) |2.93 / 2.58|3.84 / 3.63| 99.34 / 99.45 |99.37 / 99.48 |99.32 / 99.46|
|LSUN (crop) |3.42 / 3.70 |4.12 / 4.32 |99.25 / 99.27|99.29 / 99.32|99.24 / 99.26|
|LSUN (resize) |0.77 / 1.58|2.1 / 2.69 |99.75 / 99.67 |99.77 / 99.68 |99.73 / 99.68|

#### DenseNet on CIFAR-100
| OOD Dataset | FPR at 95% TPR ↓ | Detection Error ↓ | AUROC ↑ | AUPR In ↑ | AUPR Out ↑ |
|:---|:---:|:---:|:---:|:---:|:---:|
|Tiny-ImageNet (crop) |8.29 / 4.78|6.27 / 4.86|98.43 / 99.00| 98.58 / 99.05|98.3 / 99.00|
|Tiny-ImageNet (resize) |20.52 / 12.09|9.98 / 7.63| 96.27 / 97.80 |96.66 / 98.01 |95.82 / 97.66|
|LSUN (crop) |14.69 / 11.67 |8.46 / 7.58 |97.37 / 97.87|97.62 / 98.00|97.18 / 97.81|
|LSUN (resize) |16.23 / 9.30|8.77 / 6.44 |97.03 / 98.21 |97.37 / 98.42 |96.6 / 97.96|

#### WideResNet on CIFAR-10
| OOD Dataset | FPR at 95% TPR ↓ | Detection Error ↓ | AUROC ↑ | AUPR In ↑ | AUPR Out ↑ |
|:---|:---:|:---:|:---:|:---:|:---:|
|Tiny-ImageNet (crop) |0.82 / 2.08|2.24 / 3.48|99.75 / 99.55| 99.77 / 99.57|99.75 / 99.54|
|Tiny-ImageNet (resize) |2.94 / 4.29|3.83 / 4.62| 99.36 / 99.15 |99.4 / 99.17 |99.36 / 99.17|
|LSUN (crop) |1.93 / 4.65 |3.24 / 4.71 |99.55 / 99.07|99.57 /  99.17|99.55 / 99.02|
|LSUN (resize) |0.88 / 3.25|2.52 / 3.92 |99.7 / 99.39 |99.72 / 99.37 |99.68 / 99.43|

#### WideResNet on CIFAR-100
| OOD Dataset | FPR at 95% TPR ↓ | Detection Error ↓ | AUROC ↑ | AUPR In ↑ | AUPR Out ↑ |
|:---|:---:|:---:|:---:|:---:|:---:|
|Tiny-ImageNet (crop) |9.17 / 8.75|6.67 /  6.51|98.22 / 98.30| 98.39 / 98.48|98.07 / 98.11|
|Tiny-ImageNet (resize) |24.53 / 21.09|11.64 / 10.00| 95.18 / 96.13 |95.5 / 96.63 |94.78 / 95.27|
|LSUN (crop) |14.22 / 15.55 |8.2 / 8.51 |97.38 / 97.25|97.62 /  97.49|97.16 / 97.09|
|LSUN (resize) |16.53 / 15.16|9.14 / 7.95 |96.77 /  97.19 |97.03 / 97.64 |96.41 / 96.35|


## References
- [1]: A. Vyas, N. Jammalamadaka and et al. "Out-of-Distribution Detection Using an Ensemble of Self Supervised Leave-out Classifiers", in ECCV, 2018.
- [2]: S. Liang, Y. Li and R. Srikant. "Enhancing The Reliability of Out-of-distribution Image Detection in Neural Networks", in ICLR, 2018.
