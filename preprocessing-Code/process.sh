#!/bin/sh
TFs=( $(awk '{print $1}' xaa) )
    echo calulating coverage ...
    for j in "${TFs[@]}"; do
    i=${j%".bam"}
	echo $i;
    ~/samtools-1.9/samtools index $i.bam > $i.bam.bai
	~/bedtools2/bin/bedtools multicov -bams $i.bam -bed GeneFile.windows.bed | awk '{print $5}'> $i.multicov.txt;

	sed -i -e '1i'$i'\' $i.multicov.txt;
     
done
#paste -d"" geneindex $i.multicov.txt > data.txt
