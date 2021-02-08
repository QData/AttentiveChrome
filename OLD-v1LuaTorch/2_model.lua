----------------------------------------------------------------------
-- NN model

---------------------------------------------------------------------
require 'torch'   -- torch
require 'image'   -- for image transforms
require 'nn'      -- provides all sorts of trainable modules/layers
require 'math'
require 'torch'
require 'nn'
require './util/LSTM'
require './util/ReverseSequence'

print '==>AttentiveChrome Classification Model' 
print '==> define parameters'

-- Classification problem
  noutputs = 2

  rnn_size = opt.rnn_size
  filtsize = opt.cnn_size
  poolsize = opt.cnn_pool
 
-- input dimensions
nfeats=5
width = 100
ninputs = nfeats*width


-- hidden units, filter sizes (for CNN)
nstates = {50,625,125}
padding = math.floor(filtsize/2)
----------------------------------------------------------------------

local function create_lstm(ipsize,rnn_size,len,reverse)
  local lstm = nn.Sequential()

  if reverse then lstm:add(nn.ReverseSequence(2,opt.gpu)) end
  	
  local rnn = nn.LSTM(ipsize, rnn_size)
  rnn.remember_states = false

  lstm:add(rnn) 
  lstm:add(nn.Dropout(0.5))
  if opt.model=="rnn" or opt.model=="cnn-rnn" then
     lstm:add(nn.Select(2,len))
  end

  if reverse then lstm:add(nn.ReverseSequence(2,opt.gpu)) end

  return lstm
end

-----------------------------------------------------------------------

local function create_attention_rnn(ipsize,rnn_size,len)


   local RNN = nn.Sequential()

   fwd = create_lstm(ipsize,rnn_size,len,false)
   bwd = create_lstm(ipsize,rnn_size,len,true)

   local concat = nn.ConcatTable()
   local output_size

   if opt.unidirectional=="true" then
      concat:add(fwd) --  ConcatTable for consistency w/ b-lstm
      output_size = rnn_size
      RNN:add(concat)
      RNN:add(nn.JoinTable(2))
      
   else
      concat:add(fwd)
      concat:add(bwd)
      output_size = rnn_size*2
      RNN:add(concat)
      RNN:add(nn.JoinTable(3))
   end

   

   local opsize=output_size
   
   local u_i = nn.Sequential()
   local h_i= nn.Sequential()
   
   local h_i1= nn.SplitTable(2)
   local h_i2=nn.Identity()

  
   u_i:add(h_i1)
        
   local alpha=nn.MapTable()
   alpha:add(nn.Linear(opsize,1,false))
   
   u_i:add(alpha)
   u_i:add(nn.JoinTable(2))
   u_i:add(nn.SoftMax())
   u_i:add(nn.Replicate(opsize,3))
   u_i:add(nn.Select(1,1))

   h_i:add(h_i2)
   h_i:add(nn.Select(1,1))

   s_i=nn.ConcatTable()     
   s_i:add(u_i)
   s_i:add(h_i)
   

   local attention=nn.Sequential()
   attention:add(nn.Reshape(len,ipsize))
   attention:add(RNN)
   attention:add(h_i2)
   attention:add(s_i)	
   attention:add(nn.CMulTable())
   attention:add(nn.Sum(1))
   
   return attention
end 

---------------------------------------------------------------------------





-----------------------------------------------------------------

print '==> construct model'

model = nn.Sequential()


if opt.model=="mlp" then

--Baseline model
  
	-- a typical MLP 
 
	ninputs=width*nfeats
  	model:add(nn.Reshape(ninputs))
  
  	model:add(nn.Linear(ninputs, nstates[3]))
	model:add(nn.Dropout(0.5))

  	model:add(nn.ReLU())
  	model:add(nn.Linear(nstates[3], nstates[1]))

  	model:add(nn.ReLU())
  	model:add(nn.Linear(nstates[1], noutputs))

elseif opt.model=="cnn" then

--CNN model
  
	-- a typical modern convolution network (conv+relu+pool)

  	-- stage 1 : filter bank -> squashing -> Max pooling
  	model:add(nn.TemporalConvolution(nfeats, nstates[1], filtsize))
  	model:add(nn.ReLU())
  	model:add(nn.TemporalMaxPooling(poolsize))

  	-- stage 2 : standard 2-layer neural network
  	model:add(nn.View(math.ceil((width-filtsize)/poolsize)*nstates[1]))
  	model:add(nn.Dropout(0.5))
  	model:add(nn.Linear(math.ceil((width-filtsize)/poolsize)*nstates[1], nstates[2]))


	model:add(nn.ReLU())
        model:add(nn.Linear(nstates[2], nstates[3]))

        model:add(nn.ReLU())
        model:add(nn.Linear(nstates[3], noutputs))

elseif opt.model=="rnn" then

-- RNN model

       local RNN = nn.Sequential()

       fwd = create_lstm(nfeats,rnn_size,width,false)
       bwd = create_lstm(nfeats,rnn_size,width,true)

       local concat = nn.ConcatTable()
       local output_size

       if opt.unidirectional=="true" then
       	  concat:add(fwd) --  ConcatTable for consistency w/ b-lstm
      	  output_size = rnn_size
       else
          concat:add(fwd)
      	  concat:add(bwd)
          output_size = rnn_size*2
       end

       RNN:add(concat)
       RNN:add(nn.JoinTable(2))

       model:add(RNN)
       
       model:add(nn.Linear(output_size, noutputs))


elseif opt.model=="rnn-attention" then

-- RNN-Attention model

       model:add(create_attention_rnn(nfeats,rnn_size,width))
       
       model:add(nn.Linear(1, noutputs))


elseif opt.model=="rnn-hie-attention" then

-- RNN-Hierarchical Attention model
       
       local word=nn.Parallel(3,1) 	
       for i=1,nfeats,1 do
       	   word:add(create_attention_rnn(1,rnn_size,width))
       end
       
       model:add(word)
       model:add(nn.View(nfeats*rnn_size*2))
       model:add(nn.Linear(nfeats*rnn_size*2, noutputs))
  
end


function model_resetStates()

	 if opt.unidirectional=="true" then
 	    fwd:get(1):resetStates()
       	 else
	    fwd:get(1):resetStates()
	    bwd:get(2):resetStates()
         end
end
	 


----------------------------------------------------------------------
print '==> here is the model:'
print(model)

----------------------------------------------------------------------

