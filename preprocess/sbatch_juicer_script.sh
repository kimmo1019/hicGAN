#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --mem=10000
#SBATCH --error=./hic_juicer_error.txt
#SBATCH --output=./hic_juicer_output.txt



ratio=16
chrom=$1
DPATH=$2
CELL=$3
resolution=$4
juicer_tool=$5
java -jar $juicer_tool dump observed NONE $DPATH/$2/total_merged.hic $1 $1 BP $resolution $DPATH/$2/intra_NONE/chr$1_10k_intra_NONE.txt 
java -jar $juicer_tool dump observed VC $DPATH/$2/total_merged.hic $1 $1 BP $resolution $DPATH/$2/intra_VC/chr$1_10k_intra_VC.txt 
java -jar $juicer_tool dump observed KR $DPATH/$2/total_merged.hic $1 $1 BP $resolution $DPATH/$2/intra_KR/chr$1_10k_intra_KR.txt 

java -jar $juicer_tool dump observed NONE $DPATH/$2/total_merged_downsample_ratio_$ratio.hic $1 $1 BP $resolution $DPATH/$2/intra_NONE/chr$1_10k_intra_NONE_downsample_ratio$ratio.txt
java -jar $juicer_tool dump observed VC $DPATH/$2/total_merged_downsample_ratio_$ratio.hic $1 $1 BP $resolution $DPATH/$2/intra_VC/chr$1_10k_intra_VC_downsample_ratio$ratio.txt
java -jar $juicer_tool dump observed KR $DPATH/$2/total_merged_downsample_ratio_$ratio.hic $1 $1 BP $resolution $DPATH/$2/intra_KR/chr$1_10k_intra_KR_downsample_ratio$ratio.txt

