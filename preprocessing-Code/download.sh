desired_factor=(H3K27me3 H3K4me3 H3K9me3 H3K4me2)
encode_file_accession=(ENCFF000VDB ENCFF236SNL ENCFF000BYT ENCFF000BXS)
for i in ${!desired_factor[@]}
do
	echo ${desired_factor[$i]}
wget -O ${desired_factor[$i]}_${encode_file_accession[$i]}.bam https://www.encodeproject.org/files/${encode_file_accession[$i]}/@@download/${encode_file_accession[$i]}.bam
done