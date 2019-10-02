# hicGAN
We proposed hicGAN, an open-sourced framework, for inferring high resolution Hi-C data from low resolution Hi-C data with generative adversarial networks (GANs)

This work has been presented in ISMB2019 conference in an oral talk during July 21-25, Switzerland.

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

**Step 1**: Download raw aligned sequencing reads from Hi-C experiments

We preprocess Hi-C data from alighed sequencing reads (e.g. ```GSM1551550_HIC001_merged_nodups.txt.gz``` from Rao *et al*. 2014). One can directly download raw Hi-C data from GEO database or refer to our `raw_data_download_script.sh` script in the `preprocess` folder. You will generate raw Hi-C data under a `PATH-to-hicGAN/data/CELL` folder. Please note that the download may take long time.

**Step 2**: Generate Hi-C raw contacts for both high resolutio Hi-C data and down-sampled low resolution Hi-C data given a 
resolution

We use Juicer toolbox for preprocessing the raw Hi-C data. Ensure that `Java` and `Juicer toolbox` are installed in your system. One can generate Hi-C raw contacts for both high resolutio Hi-C data and down-sampled low resolution Hi-C data by running `preprocess.sh` script in the `preprocess` folder. Note that one can speed up the preprocessing using `slurm` by modify one line of `preprocess.sh`. See annotation in `preprocess.sh`.
```shell
bash preprocess.sh <CELL> <Resolution> <path/to/juicer_tools.jar>
```
For example, one can directly run `bash preprocess.sh GM12878 10000 path/to/juicer_tools.jar` to extract Hi-C raw contacts of GM12878 cell line with resolution 10k.


**Step 3**: Preprate the training and test data

Typically, Hi-C samples from chromosomes 1-17 will be kept for training and chromosomes 18-22 will be kept for testing in each cell type.

```shell
python data_split.py  <CELL>
```
For example, one can directly run `python data_split.py GM12878` to generate `train_data.hkl` and `test_data.hkl` under the `data/GM12878 data folder`. 

**Step 4**: Run hicGAN model

After preparing the training and test data, one can run the following commond to run hicGAN
```shell
python run_hicGAN.py <GPU_ID> <checkpoint> <graph> <CELL>
```
For example, one can run `python run_hicGAN Checkpoint/GM12878 log/GM12878 Graph/GM12878 GM12878` 
Note that `checkpoint` is the folder to save model and 'graph' is the folder for visualization with `TensorBoard` and `log` is the folder to save the loss during the training process. The three folders will be created if not exist.

**Step 5**: Evaluate hicGAN model

After model training, one can evaluate the hicGAN by calculating MSR, PSNR and SSIM measurements, just run the following commond
```shell
python hicGAN_evaluate.py <GPU_ID> <MODEL_PATH> <CELL>
```
For example, one can run `python hicGAN_evaluate.py 0 checkpoint GM12878` for model evaluation.

We finally provide a `demo.ipynb` to illustrate the above steps with a demo of Hi-C model.

We also provide a `Results_reproduce` to show how the results in our paper were produced.

Note that we also provide a pre-trained model of hicGAN which was trained in GM12878 cell line.

# Run hicGAN on your own data
We provided instructions on implementing hicGAN model from raw aligned sequencing reads. One could directly run hicGAN model with custom data by constructing low resolution data and corresponding high resolution data in `run_hicGAN.py` with custom data by the following instructions. 

**Step 1**: Modify one line in `run_hicGAN.py`

You can find `lr_mats_train_full, hr_mats_train_full = hkl.load(...)` in `run_hicGAN.py`. All you need to do is to generate `lr_mats_train_full` and `hr_mats_train_full` by yourself. 

Note that `hr_mats_train_full` and `lr_mats_train_full` are high resolution Hi-C training samples and low resolution Hi-C training samples, respectively. The size of `hr_mats_train_full` and `lr_mats_train_full` are (nb_train,40,40,1) and (nb_train,40,40,1). 

We extracted training examples in the original Hi-C matrices by cropping non-overlaping 40 by 40 squares (resolution: 10k bp) within 2M bp. See details in `data_split.py` if necessary. 

**Step 2**: Modify one line in `hicGAN_evaluate.py`

After model training, the trained model will be saved under the `checkpoint` folder. Next, one should also modify one line in `hicGAN_evaluate.py`, `lr_mats_test,hr_mats_test,_ = hkl.load(...)`. One should generate low resolution test data(`lr_mats_test`) and high resolution test data(`hr_mats_test`) by there own.

Then, one should run `hicGAN_evaluate.py`. Note that it is recommended to choose the best model for `hicGAN_g_model` (e.g. g_hicgan_300_best.npz).

**Step 3**: Check the predicted outcome

After model evaluating, the predicted outcome will be saved in `data/CELL/hicGAN_predicted.npz` which should be the same size as `hr_mats_test`. One could use `np.load(...)` for loading the predicted data as numpy arrays.

Feel free to contact `liu-q16@mails.tsinghua.edu.cn` if you have any problem in implementing your own hicGAN model.

# Citation
**Liu Q**, Lv H, Jiang R. hicGAN infers super resolution Hi-C data with generative adversarial networks[J]. Bioinformatics, 2021, 35(14): i99-i107.
```
@article{liu2019hicgan,
  title={hicGAN infers super resolution Hi-C data with generative adversarial networks},
  author={Liu, Qiao and Lv, Hairong and Jiang, Rui},
  journal={Bioinformatics},
  volume={35},
  number={14},
  pages={i99--i107},
  year={2019},
  publisher={Oxford University Press}
}
```

# License
This project is licensed under the MIT License - see the LICENSE.md file for details
