# Attending to Discriminative Certainty for Domain Adaptation(CADA)

Torch code for Domain Adaptation model(CADA) . For more information, please refer the [paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Kurmi_Attending_to_Discriminative_Certainty_for_Domain_Adaptation_CVPR_2019_paper.pdf) 

Accepted at [[CVPR 2019](http://cvpr2019.thecvf.com/)]

#####  [[Project]](https://delta-lab-iitk.github.io/CADA//)     [[Paper Link ]](https://arxiv.org/abs/1906.03502)

#### Abstract 
In this paper, we aim to solve for unsupervised domain adaptation of classifiers where we have access to label information for the source domain while these are not available for a target domain. While various methods have been proposed for solving these including adversarial discriminator based methods, most approaches have focused on the entire image based domain adaptation. In an image, there would be regions that can be adapted better, for instance, the foreground object may be similar in nature. To obtain such regions, we propose methods that consider the probabilistic certainty estimate of various regions and specify focus on these during classification for adaptation. We observe that just by incorporating the probabilistic certainty of the discriminator while training the classifier, we are able to obtain state of the art results on various datasets as compared against all the recent methods. We provide a thorough empirical analysis of the method by providing ablation analysis, statistical significance test, and visualization of the attention maps and t-SNE embeddings. These evaluations convincingly demonstrate the effectiveness of the proposed approach

![Result](https://delta-lab-iitk.github.io/CADA/cada/model_cada.png) 


### Requirements
This code is written in Lua and requires [Torch](http://torch.ch/). 


You also need to install the following package in order to sucessfully run the code.
- [Torch](http://torch.ch/docs/getting-started.html#_)
- [loadcaffe](https://github.com/szagoruyko/loadcaffe)


#### Download Dataset
- [Office -31](https://pan.baidu.com/s/1o8igXT4)
- [ImageClef](https://pan.baidu.com/s/1lx2u1SMlSamsHnAPWrAHWA)
- [Office-Home](http://hemanthdv.org/OfficeHome-Dataset/)

##### Prepare Datasets
- Download the dataset


### Training Steps

We have prepared everything for you ;)

####Clone the repositotry 

``` git clone https://github.com/DelTA-Lab-IITK/CADA  ```

#### Dataset prepare
- Download dataset

-  put all source images inside mydataset/train/ such that folder name is class name
```
  mkdir -p /path_to_wherever_you_want/mydataset/train/ 
```
- put all target images inside mydataset/val/ such that folder name is class name

``` 
mkdir -p /path_to_wherever_you_want/mydataset/val/ 
```
- creare softlink of dataset
```
 cd CADA/
 ln -sf /path_to_wherever_you_want/mydataset dataset
```
 
  

#### Pretrained Alexnet model
- Download Alexnet pretrained caffe model [Link](https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet)

#### Pretrained ResNet model
- Download ResNet pretrained  model [Link](https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet)

``` 
cd CADA/  
```

```
ln -sf /path_to_where_model_is_downloaded/ pretrained_network 
```

#### Train model
``` 
cd CADA/  
./train.sh 
```




### Reference

If you use this code as part of any published research, please acknowledge the following paper

```
@InProceedings{Kurmi_2019_CVPR,
author = {Kumar Kurmi, Vinod and Kumar, Shanu and Namboodiri, Vinay P.},
title = {Attending to Discriminative Certainty for Domain Adaptation},
booktitle = {IEEE Computer Society Conference on Computer Vision and Pattern Recognition(CVPR),},
month = {June},
year = {2019}
}
```

## Contributors
* [Vinod K. Kurmi][1] (vinodkk@iitk.ac.in)
* [Shanu Kumar][2] (sshanukr@gmail.com)



[1]: https://github.com/vinodkkurmi
[2]: https://github.com/sshanu




