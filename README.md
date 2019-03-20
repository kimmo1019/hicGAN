# hicGAN
We proposed hicGAN, an open-sourced framework, for inferring high resolution Hi-C data from low resolution Hi-C data with generative adversarial networks (GANs)

![model](https://github.com/kimmo1019/hicGAN/blob/master/model.png)

hicGAN consists of two networks that compete with each other. G tries to generate super resolution samples that are highly similar to real high resolution samples while D tries to discriminate generated super resolution samples from real high resolution Hi-C samples.

# Requirements
- TensorFlow >= 1.10.0
- TensorLayer >= 1.9.1
- hickle >= 2.1.0
- Java JDK >= 1.8.0
- Juicer Tool

# Installation
hicGAN can be downloaded by
```shell
git clone https://github.com/kimmo1019/hicGAN
```
Installation has been tested in a Linux/MacOS platform.

# Instructions
We provide detailed step-by-step instructions for running hicGAN model for reproducing the results in the original paper and inferring high resolution Hi-C data of your own interst.

Step 1: Download raw aligned sequencing reads from Hi-C experiments

We preprocess Hi-C data from alighed sequencing reads (e.g. ```GSM1551550_HIC001_merged_nodups.txt.gz``` from Rao *et al*. 2014). One can directly download raw Hi-C data from GEO database or refer to our `raw_data_download_script.sh` script in the `preprocess` folder. Prepare your raw Hi-C data under a `CELL` folder.

Step 2: Generate Hi-C raw contacts for both high resolutio Hi-C data and down-sampled low resolution Hi-C data given a 
resolution

We use Juicer toolbox for preprocessing the raw Hi-C data. Ensure that `Java` and `Juicer toolbox` are installed in your system. One can generate Hi-C raw contacts for both high resolutio Hi-C data and down-sampled low resolution Hi-C data by running `preprocess.sh` script in the `preprocess` folder.
```shell
bash preprocess.sh <PATH-TO-DATA> <CELL> <Resolution>
```
For example, one can directly run `bash preprocess.sh data data/GM12878 10000` to extract Hi-C raw contacts with resolution 10k.

Step 3: Preprate the training and test data

Typically, Hi-C samples from chromosomes 1-17 will be kept for training and chromosomes 18-22 will be kept for testing in each cell type.

```shell
python data_split.py <PATH-TO-PREPROCESSED-DATA> <PATH-TO-SAVA-DATA>
```
For example, one can directly run `python data_split.py data/GM12878 data/GM12878/train_test_split` to generate `train_data.hkl` and `test_data.hkl` 

Step 4: Run hicGAN model
After preparing the training and test data, one can run the following commond to run hicGAN
```shell
python run_hicGAN.py <gpu_id> <checkpoint> <graph> <PATH-TO-SAVA-DATA>
```
Note that `checkpoint` is the folder to save model and 'graph' is the folder for visualization with `TensorBoard`.

We finally provide a `demo.ipynb` to implement the above steps with a demo of Hi-C model.

Note that we also provide a pre-trained model of hicGAN which was trained in K562 cell line.




# License
This project is licensed under the MIT License - see the LICENSE.md file for details
