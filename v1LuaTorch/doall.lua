----------------------------------------------------------------------
-- This script performs training and testing on 
-- AttentiveChrome model on a gene expression
-- classification problem. 
-- Ritambhara Singh
----------------------------------------------------------------------
require 'torch'

---------------------------------------------------------------------
print '==> processing options'

local set = function()
      cmd = torch.CmdLine()
      cmd:text()
      cmd:text('AttentiveChrome Pipeline options')
      cmd:text()
      cmd:text('Options:')

      --GPU
      cmd:option('-gpu', -1, 'set >=0 (GPU number) to run with CUDA on GPU')

      -- global:
      cmd:option('-seed', 1, 'fixed input seed for repeatable experiments')
      cmd:option('-threads', 2, 'number of threads')
      cmd:option('--epochs', 10, 'number of epochs')
      
      --data:
      cmd:option('-dataDir', "data/", 'The data home location')
      cmd:option('-dataset', "toy/", 'Dataset name, corresponds to the folder name in dataDir')
      cmd:option('-resultsDir', "results/", 'The data home location')
      cmd:option('-name', "", 'Optionally, give a name for this model')
      cmd:option('-trsize', "10", 'Training set size (number of genes)')
      cmd:option('-tssize', "10", 'Test set size (number of genes)')

      --model:
      cmd:option('-nonlinearity', 'relu', 'type of nonlinearity function to use: tanh | relu | prelu')
      cmd:option('-loss', 'nll', 'type of loss function to minimize: nll')
      cmd:option('-model', 'rnn-hie-attention', 'type of models : mlp|cnn|rnn|rnn-attention|rnn-hie-attention')
      
      cmd:option('-cnn_size', 10, 'Window size of mlp or convolution kernel sizes of conv')
      cmd:option('-cnn_pool', 5 , 'sizes of pooling windows')
      cmd:option('-rnn_size', 32, 'hidden layer size for RNN')
      cmd:option('-unidirectional', 'false', 'for selecting uni/bi-directional LSTM')
      

      -- training:
      cmd:option('-save', 'results', 'subdirectory to save/log experiments in')
      cmd:option('-plot', false, 'live plot')
      cmd:option('-learningRate', 1e-3, 'learning rate at t=0')
      cmd:option('-batchSize', 1, 'mini-batch size (1 = pure stochastic), currently the model does not support >1')
      cmd:option('-weightDecay', 0, 'weight decay (SGD only)')
      cmd:option('-momentum', 0, 'momentum (SGD only)')
      cmd:option('-grad_clip',5, 'threshold for clipping gradients')


      cmd:text()
      opt = cmd:parse(arg or {})
end

set()

if opt.gpu >= 0  then
  collectgarbage()
  require 'cutorch'
  require 'cunn'
  cutorch.setDevice(opt.gpu + 1)
  dtype = 'torch.CudaTensor'
  itype = 'cuda()'
  print(string.format('Running with CUDA on GPU %d', opt.gpu+1))
else
  dtype = 'torch.FloatTensor'
  itype = 'float()'
  print 'Running in CPU mode'
end

torch.setnumthreads(opt.threads)
torch.manualSeed(opt.seed)

----------------------------------------------------------------------
print '==> executing all'
dofile '1_data.lua'
dofile '2_model.lua'
dofile '3_loss.lua'
dofile '4_train.lua'
dofile '5_test.lua'

----------------------------------------------------------------------
print '==> training!'
local i = 1

while i<opt.epochs do
   train()
   test()
   i=i+1
end
