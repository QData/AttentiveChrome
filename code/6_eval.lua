----------------------------------------------------------------------
-- This script implements a test procedure, to report accuracy
-- on the test data.
--
-- Ritambhara Singh
----------------------------------------------------------------------

require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods
include('util/auRoc.lua')
require './util/LSTM'
require './util/ReverseSequence'
require 'lfs'
require 'nn'
----------------------------------------------------------------------
print '==> loading data'

width=100
nfeats=5

testdataset={}
function testdataset:size() return opt.tssize end

for i=1,testdataset:size() do
  local input = testset[i].data;     
  local output = testset[i].label;
  testdataset[i] = {input, output}
end

print '==> defining some tools'

--classes

classes = {'1','2'}

filePath=opt.resultsDir .. opt.dataset

print '==> defining test procedure'
local AUC = auRoc:new()
local AUC_target=1

-- Log results to files
testLogger = optim.Logger(paths.concat(filePath,opt.model .. '.test-best.log'))

print '==> loading model'
epoch=opt.epoch
local modelname = ('model.' .. opt.model .. '.' .. opt.epoch .. '.net')
local filename = paths.concat(filePath, modelname)
model = torch.load(filename)

-- test function
function test()
   -- local vars
   local time = sys.clock()

   -- averaged param use?
   if average then
      cachedparams = parameters:clone()
      parameters:copy(average)
   end

   -- set model to evaluate mode (for modules that differ in training and testing, like Dropout)
   model:evaluate()

   -- test over test data
   print('==> testing on test set:')
   for t = 1,testdataset:size(),opt.batchSize do
      -- disp progress
      xlua.progress(t, testdataset:size())

      -- get new sample
      local input = testdataset[t][1]:type(dtype)
      local target = testdataset[t][2]

      -- test sample
      input_reshape=torch.reshape(input,opt.batchSize,width,nfeats):type(dtype)
      local pred = model:forward(input_reshape)

      score=torch.reshape(pred,opt.batchSize,2):type(dtype)

      auc_in = score[{{1,score:size(1)},{1,1}}]:reshape(score:size(1))
      for i = 1,auc_in:size(1) do 
	   if target == 2 then AUC_target=-1 else AUC_target=1 end
	   AUC:add(math.exp(auc_in[i]), AUC_target)
      end

      
   end

   -- timing
   time = sys.clock() - time
   time = time / testdataset:size()
   print("\n==> time to test 1 sample = " .. (time*1000) .. 'ms')

   --print AUC score
   local AUROC = AUC:calculateAuc()
   print(' + AUROC: '..AUROC)
  

   -- update log
   testLogger:add{['AUC Score (test set)'] = AUROC}
   

   -- averaged param use?
   if average then
      -- restore parameters
      parameters:copy(cachedparams)
   end
   
   -- next iteration:
   AUC:zero()
end
