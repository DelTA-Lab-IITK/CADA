
--- Taken from https://github.com/soumith/dcgan.torch/blob/master/data/data.lua
-- Updated by Vinod Kumar Kurmi(vinodkumarkurmi@gmail.com)
--[[
    This data loader is a modified version of the one from dcgan.torch
    (see https://github.com/soumith/dcgan.torch/blob/master/data/data.lua).
    
    Copyright (c) 2016, Deepak Pathak [See LICENSE file for details]
]]--

local Threads = require 'threads'
Threads.serialization('threads.sharedserialize')

local data = {}

local result = {}
local unpack = unpack and unpack or table.unpack

function data.new(n, opt_)
   opt_ = opt_ or {}
   local self = {}
   for k,v in pairs(data) do
      self[k] = v
   end
    n=0
   local donkey_file = 'donkey_folder.lua'
   if n > 0 then
      local options = opt_
      self.threads = Threads(n,
                             function() require 'torch' end,
                             function(idx)
                                opt = options
                                tid = idx
                                local seed = (opt.manual_seed and opt.manual_seed or 0) 
                                torch.manualSeed(seed)
                                torch.setnumthreads(1)
                                print(string.format('Starting donkey with id: %d seed: %d', tid, seed))
                                assert(options, 'options not found')
                                assert(opt, 'opt not given')
                                print(opt)
                                paths.dofile(donkey_file)
                             end
      )
   else
      if donkey_file then paths.dofile(donkey_file) end
      self.threads = {}
      function self.threads:addjob(f1, f2) f2(f1()) end
      function self.threads:dojob() end
      function self.threads:synchronize() end
   end

   local nSamples = 0
   self.threads:addjob(function() return testLoader:size() end,
         function(c) nSamples = c end)
   self.threads:synchronize()
   self._size = nSamples

   for i = 1, n do
      self.threads:addjob(self._getFromThreads,
                          self._pushResult)
   end

   return self
end

function data._getFromThreads()
   assert(opt.Test_batchSize, 'opt.Test_batchSize not found')
   return testLoader:sampleSequence( opt.Test_batchSize,opt.start_Batch_IndexTest)
end

function data._pushResult(...)
   local res = {...}
   if res == nil then
      self.threads:synchronize()
   end
   result[1] = res
end



function data:getBatch(start_Batch_IndexTest,test_data_size)

    opt.Test_batchSize=math.min(test_data_size-start_Batch_IndexTest+1,opt.Test_batchSize)
    opt.start_Batch_IndexTest=start_Batch_IndexTest
   -- queue another job
    
   self.threads:addjob(self._getFromThreads, self._pushResult)
   self.threads:dojob()

   local res = result[1]
   result[1] = nil
   if torch.type(res) == 'table' then
      return unpack(res)
   end

   print(type(res))

   return res
end

function data:size()
   return self._size
end

return data
