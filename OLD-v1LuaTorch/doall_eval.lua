----------------------------------------------------------------------
-- This script performs training and testing for evaluation/visualization on 
-- AttentiveChrome model on gene expression 
-- classification problem. 
--

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
      cmd:option('-epoch', 1, 'epoch of the best trained model')
      --data:
      cmd:option('-dataDir', "data/", 'The data home location')
      cmd:option('-dataset', "toy/", 'Dataset name, corresponds to the folder name in dataDir')
      cmd:option('-resultsDir', "results/", 'The data home location')
      cmd:option('-name', "", 'Optionally, give a name for this model')
      cmd:option('-tssize', "9", 'Test set size (number of genes)')

      --model:
      cmd:option('-model', 'rnn-hie-attention', 'type of models : mlp|cnn|rnn|rnn-attention|rnn-hie-attention')
      cmd:option('-rnn_size', 32, 'hidden layer size for RNN')
      
      cmd:option('-batchSize', 1, 'mini-batch size (1 = pure stochastic), currently the model does not support >1')
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
dofile '1_data_eval.lua'
--perform testing:
dofile '6_eval.lua'

if opt.model=="rnn-hie-attention" then
    --perform alpha-attention visualization
    dofile '7_viz_alpha.lua'
    --perform beta-attention visualization
    dofile '8_viz_beta.lua'
end
----------------------------------------------------------------------
print '==> evaluating!'
test()

if opt.model=="rnn-hie-attention" then
    print '==> printing alpha attention map!'
    viz_alpha()

    print '==> printing beta attention map!'
    viz_beta()
end
