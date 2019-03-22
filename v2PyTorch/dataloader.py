import torch
import collections
import pdb
import csv
from kipoi.data import Dataset
import math
import numpy as np

class HMData(Dataset):
	# Dataset class for loading data
	def __init__(self, input_file, expr_file, bin_size=200):
		self.gene_dict = self.loadDict(expr_file)
		self.threshold = np.median(np.array(list(self.gene_dict.values())))
		self.hm_data = self.loadData(input_file, bin_size, self.gene_dict, self.threshold)


	def loadDict(self, filename):
		# get expression value of each gene from cell*.expr.csv
		gene_dict={}
		with open(filename) as fi:
			for line in fi:
				geneID,geneExpr=line.split(',')
				gene_dict[str(geneID)]=float(geneExpr)
		fi.close()
		return(gene_dict)

	def loadData(self,filename, windows, gene_dict, threshold):
		with open(filename) as fi:
			csv_reader=csv.reader(fi)
			data=list(csv_reader)

			ncols=(len(data[0]))
		fi.close()
		nrows=len(data)
		ngenes=nrows/windows
		nfeatures=ncols-1

		count=0
		attr=collections.OrderedDict()

		for i in range(0,nrows,windows):
			hm1=torch.zeros(windows,1)
			hm2=torch.zeros(windows,1)
			hm3=torch.zeros(windows,1)
			hm4=torch.zeros(windows,1)
			hm5=torch.zeros(windows,1)
			for w in range(0,windows):
				hm1[w][0]=int(data[i+w][1])
				hm2[w][0]=int(data[i+w][2])
				hm3[w][0]=int(data[i+w][3])
				hm4[w][0]=int(data[i+w][4])
				hm5[w][0]=int(data[i+w][5])
			geneID=str(data[i][0].split("_")[0])

			# stop()
			if gene_dict[geneID] >= threshold:
				thresholded_expr = 1
			else:
				thresholded_expr = 0
			attr[count]={
				'geneID':geneID,
				# 'expr':gene_dict[geneID],
				'expr':thresholded_expr,
				'hm1':hm1,
				'hm2':hm2,
				'hm3':hm3,
				'hm4':hm4,
				'hm5':hm5
			}
			count+=1

		return attr

	def getlabel(self, c1):
		# get log fold change of expression

		label1=math.log((float(c1)+1.0),2)
		label=[]
		label.append(label1)

		fold_change=(float(c1)+1.0)/(float(c1)+1.0)
		log_fold_change=math.log((fold_change),2)
		return (log_fold_change, label)

	def __len__(self):
		return len(self.hm_data)

	def __getitem__(self,i):
		final_data=torch.cat((self.hm_data[i]['hm1'],self.hm_data[i]['hm2'],self.hm_data[i]['hm3'],self.hm_data[i]['hm4'],self.hm_data[i]['hm5']),1)
		#final_data = final_data.view(-1,final_data.size(0),final_data.size(1))
		final_data = final_data.numpy()
		label,orig_label=self.getlabel(self.hm_data[i]['expr'])
		b_label_c1=orig_label[0]
		geneID=self.hm_data[i]['geneID']


		return_item={'inputs': final_data,
					'metadata': {'geneID':geneID,'diff':label, 'abs_A':b_label_c1}
					}

		return return_item
