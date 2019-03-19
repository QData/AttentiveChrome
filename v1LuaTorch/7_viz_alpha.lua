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

-- removing layers to print out alpha attention layer output 
alpha_model=model:clone()  

alpha_model:remove(4)
alpha_model:remove(3)
alpha_model:remove(2)

m1=alpha_model.modules[1]


for i=1,nfeats,1 do

    m2=m1.modules[i]
    m2:remove(6)
    m2:remove(5)

    m3=m2.modules[4]
    m3.modules[2]=nil
    m4=m3.modules[1]
    m4:remove(6)
    m4:remove(5)
    m2.modules[4]=nil
    m2:add(m4)
end

print(alpha_model)

-- test function
function viz_alpha()
   -- local vars
   -- local time = sys.clock()

   -- averaged param use?
   if average then
      cachedparams = parameters:clone()
      parameters:copy(average)
   end

   -- set model to evaluate mode (for modules that differ in training and testing, like Dropout)
   model:evaluate()
   alpha_model:evaluate()

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


      local alpha = alpha_model:forward(input_reshape)
      -- print alpha-attention output
      print(alpha)
      
   end

end
