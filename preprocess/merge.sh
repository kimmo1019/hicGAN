#merge HIC001-HIC018
find *.txt.gz | xargs zcat | sort -k3,3d -k7,7d > total_merged_nodups.txt
