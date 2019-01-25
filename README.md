# hicGAN
We proposed hicGAN, an open-sourced framework, for inferring high resolution Hi-C data from low resolution Hi-C data with generative adversarial networks (GANs)

![model](https://github.com/kimmo1019/hicGAN/blob/master/model.png)

hicGAN consists of two networks that compete with each other. G tries to generate super resolution samples that are highly similar to real high resolution samples while D tries to discriminate generated super resolution sam- ples from real high resolution Hi-C samples.

# Requirements
- TensorFlow >= 1.10.0
- TensorLayer >= 1.9.1
- hickle >= 2.1.0

# Installation
Download hicGAN by
```shell
git clone https://github.com/kimmo1019/hicGAN
```
Installation has been tested in a Linux/MacOS platform with Python2.7.

# Instructions



# License
This project is licensed under the MIT License - see the LICENSE.md file for details
