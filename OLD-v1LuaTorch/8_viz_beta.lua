----------------------------------------------------------------------
-- This script implements a test procedure, to report accuracy
-- on the test data.
--
-- Ritambhara Singh
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


print '==> loading model'
epoch=opt.epoch
local modelname = ('model.' .. opt.model .. '.' .. opt.epoch .. '.net')
local filename = paths.concat(filePath, modelname)
model = torch.load(filename)

-- removing layers to print out beta attention layer output 
beta_model=model:clone()

beta_model:remove(4)
beta_model:remove(3)
beta_model:remove(2)
beta_model:add(nn.Reshape(nfeats,opt.rnn_size*2)):type(dtype)
beta_model:add(nn.Mean(2)):type(dtype)



print(beta_model)

-- test function
function viz_beta()
   -- local vars
   -- local time = sys.clock()

   -- averaged param use?
   if average then
      cachedparams = parameters:clone()
      parameters:copy(average)
   end

   -- set model to evaluate mode (for modules that differ in training and testing, like Dropout)
   beta_model:evaluate()

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

      local beta = beta_model:forward(input_reshape)
      -- print beta-attention output
      print(beta)
      
   end

end
