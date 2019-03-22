import numpy
import torch
import scipy 
import scipy.sparse as sp
import logging
from six.moves import xrange
from collections import OrderedDict
import sys
import pdb
from sklearn import metrics
import torch.nn.functional as F
from torch.autograd import Variable
def compute_metrics(predictions, targets):

	pred=predictions.numpy()
	targets=targets.numpy()

	R2,p=scipy.stats.pearsonr(numpy.squeeze(targets),numpy.squeeze(pred))
	MSE=metrics.mean_squared_error(targets, pred)
	return MSE, R2



