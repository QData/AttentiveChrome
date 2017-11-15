----------------------------------------------------------------------
-- This script defines training procedure,
-- irrespective of the model/loss functions chosen.
--
-- It is used to 

--   + define a closure to estimate (a noisy) loss
--     function, as well as its derivatives wrt the parameters of the
--     model to be trained
--   + optimize the function, according to: SGD
--
-- Ritambhara Singh
----------------------------------------------------------------------

require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods
include('util/auRoc.lua')
require 'nn'
require 'lfs'
----------------------------------------------------------------------

print '==> loading data'

traindataset={}
function traindataset:size() return opt.trsize end

for i=1,traindataset:size() do
  local input = trainset[i].data;    
  local output = trainset[i].label;
  traindataset[i] = {input, output}
end

testdataset={}
function testdataset:size() return opt.tssize end

for i=1,testdataset:size() do
  local input = testset[i].data;     
  local output = testset[i].label;
  testdataset[i] = {input, output}
end
------------------------------------------------------------------------------

print '==> defining some tools'

--classes

classes = {'1','2'}


local AUC = auRoc:new()
local AUC_target=1

filePath=opt.resultsDir .. opt.dataset

-- Log results to files
trainLogger = optim.Logger(paths.concat(filePath,opt.model .. '.train.log'))
testLogger = optim.Logger(paths.concat(filePath,opt.model .. '.test.log'))

-- Retrieve parameters and gradients:
-- this extracts and flattens all the trainable parameters of the mode
-- into a 1-dim vector

model:type(dtype)
criterion:type(dtype)


if model then
   parameters,gradParameters = model:getParameters()
end

----------------------------------------------------------------------
print '==> configuring optimizer : SGD'


optimState = {
 	   learningRate = opt.learningRate,	
	   weightDecay = opt.weightDecay,
 	   momentum = opt.momentum,
      	   learningRateDecay = 1e-7
}
optimMethod = optim.sgd

----------------------------------------------------------------------
print '==> defining training procedure'

function train()

   -- epoch tracker
   epoch = epoch or 1

   -- local vars
   local time = sys.clock()

   -- set model to training mode (for modules that differ in training and testing, like Dropout)

   if opt.model=="rnn" or opt.model=="rnn-attention" then model_resetStates() end
   model:training()

   -- training size
   trsize=traindataset:size()

   -- shuffle at each epoch
   shuffle = torch.randperm(trsize)

   -- do one epoch
   print('==> doing epoch on training data:')

   for t=1,trsize,opt.batchSize do
      -- disp progress
      xlua.progress(t, traindataset:size())
      
      if opt.model=="rnn" or opt.model=="rnn-attention" then model_resetStates() end
      local input = traindataset[shuffle[t]][1]:type(dtype)
      local target = traindataset[shuffle[t]][2]
  
      -- create closure to evaluate f(X) and df/dX
      local feval = function(x)
   	 -- get new parameters
         if x ~= parameters then
            parameters:copy(x)
         end

	 -- reset gradients
   	 gradParameters:zero()
 
	
       	 -- estimate f
	 input_reshape=torch.reshape(input,opt.batchSize,width,nfeats):type(dtype)
       	 local output = model:forward(input_reshape)
	 --print(output)

       	 f = criterion:forward(output,target)
	
       	 
       	 -- estimate df/dW
       	 local df_do = criterion:backward(output,target)
	
       	 model:backward(input_reshape, df_do)

	 -- clip gradients
          if opt.grad_clip > 0 then
            gradParameters:clamp(-opt.grad_clip, opt.grad_clip)
          end

	 score=torch.reshape(output,opt.batchSize,2):type(dtype)

	 auc_in = score[{{1,score:size(1)},{1,1}}]:reshape(score:size(1))
         for i = 1,auc_in:size(1) do 
	     if target == 2 then AUC_target=-1 else AUC_target=1 end
	     AUC:add(math.exp(auc_in[i]), AUC_target)
      	 end
	 	 
  	 -- return f and df/dX
       	 return f,gradParameters
      end
      optimMethod(feval, parameters, optimState)
   end
 

   -- time taken
   time = sys.clock() - time
   time = time / traindataset:size()
   print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')


   --Calculate and print AUC score
   local AUROC = AUC:calculateAuc()
   print(' + AUROC: '..AUROC)

      -- save/log current net
   local modelname = ('model.' .. opt.model .. '.' .. epoch .. '.net')
   local filename = paths.concat(filePath, modelname)
   os.execute('mkdir -p ' .. sys.dirname(filename))
   print('==> saving model to '..filename)
   torch.save(filename, model)
  

   trainLogger:add{['AUC Score (train set)'] = AUROC}

   -- next epoch
   AUC:zero()

   epoch = epoch + 1
end

