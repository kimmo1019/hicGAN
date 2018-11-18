#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --mem=5000
#SBATCH --error=./hic_error.txt
#SBATCH --output=./hic_output.txt




#wget ftp://ftp.ncbi.nlm.nih.gov/geo/samples/GSM1551nnn/GSM15515$2/suppl/GSM15515$2_HIC0$1_merged_nodups.txt.gz
resolution=10000
#cell="K562"
#analyzeHiC ./gm12878_hic -raw -chr $1 -res $resolution > ./$1/$1_hic_mat_$resolution.txt
java -jar ../../../../juicer/scripts/juicer_tools.jar dump observed VC total_merged.hic $1 $1 BP $resolution ./intra_VC/chr$1_10k_intra_VC.txt 
java -jar ../../../../juicer/scripts/juicer_tools.jar dump observed NONE total_merged.hic $1 $1 BP $resolution ./intra_NONE/chr$1_10k_intra_NONE.txt 
java -jar ../../../../juicer/scripts/juicer_tools.jar dump observed KR total_merged.hic $1 $1 BP $resolution ./intra_KR/chr$1_10k_intra_KR.txt 

java -jar ../../../../juicer/scripts/juicer_tools.jar dump observed VC total_merged_downsample_ratio_16.hic $1 $1 BP $resolution ./intra_VC/chr$1_10k_intra_VC_downsample_ratio16.txt 
java -jar ../../../../juicer/scripts/juicer_tools.jar dump observed NONE total_merged_downsample_ratio_16.hic $1 $1 BP $resolution ./intra_NONE/chr$1_10k_intra_NONE_downsample_ratio16.txt 
java -jar ../../../../juicer/scripts/juicer_tools.jar dump observed KR total_merged_downsample_ratio_16.hic $1 $1 BP $resolution ./intra_KR/chr$1_10k_intra_KR_downsample_ratio16.txt 
