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
# $(($a + $b)) is expression;   $a\_ is the $a without "_"
java -jar $juicer_tool dump observed NONE $DPATH/$CELL/total_merged.hic $chrom $chrom BP $resolution $DPATH/$CELL/intra_NONE/chr$chrom\_res$(($resolution/1000))k_intra_NONE.txt 
java -jar $juicer_tool dump observed VC $DPATH/$CELL/total_merged.hic $chrom $chrom BP $resolution $DPATH/$CELL/intra_VC/chr$chrom\_res$(($resolution/1000))k_intra_VC.txt 
java -jar $juicer_tool dump observed KR $DPATH/$CELL/total_merged.hic $chrom $chrom BP $resolution $DPATH/$CELL/intra_KR/chr$chrom\_res$(($resolution/1000))k_intra_KR.txt 

java -jar $juicer_tool dump observed NONE $DPATH/$CELL/total_merged_downsample_ratio_$ratio.hic $chrom $chrom BP $resolution $DPATH/$CELL/intra_NONE/chr$chrom\_res$(($resolution/1000))k_intra_NONE_downsample_ratio$ratio.txt
java -jar $juicer_tool dump observed VC $DPATH/$CELL/total_merged_downsample_ratio_$ratio.hic $chrom $chrom BP $resolution $DPATH/$CELL/intra_VC/chr$chrom\_res$(($resolution/1000))k_intra_VC_downsample_ratio$ratio.txt
java -jar $juicer_tool dump observed KR $DPATH/$CELL/total_merged_downsample_ratio_$ratio.hic $chrom $chrom BP $resolution $DPATH/$CELL/intra_KR/chr$chrom\_res$(($resolution/1000))k_intra_KR_downsample_ratio$ratio.txt

