--- Taken from  https://github.com/soumith/dcgan.torch/blob/master/data/donkey_folder.lua).
-- Updated by Vinod Kumar Kurmi(vinodkumarkurmi@gmail.com)

--[[
    This data loader is a modified version of the one from dcgan.torch
    (see https://github.com/soumith/dcgan.torch/blob/master/data/donkey_folder.lua).

    Copyright (c) 2016, Deepak Pathak [See LICENSE file for details]

    Copyright (c) 2015-present, Facebook, Inc.
    All rights reserved.
    This source code is licensed under the BSD-style license found in the
    LICENSE file in the root directory of this source tree. An additional grant
    of patent rights can be found in the PATENTS file in the same directory.
]]--

require 'image'
paths.dofile('dataset_target.lua')

-- This file contains the data-loading logic and details.
-- It is run by each data-loader thread.
------------------------------------------
-------- COMMON CACHES and PATHS
-- Check for existence of opt.data
opt.data = os.getenv('DATA_ROOT')
if not paths.dirp(opt.data) then
    error('Did not find directory: ', opt.data)
end

-- a cache file of the validation metadata (if doesnt exist, will be created)
local cache = "cache"
local cache_prefix = opt.data:gsub('/', '_')
os.execute('mkdir -p cache')
local valCache = paths.concat(cache, cache_prefix .. '_valCache.t7')
local trainCache = paths.concat(cache, cache_prefix .. '_trainCache.t7') -- Load for class consistent means source class=target class (load the Valclass ()no of val class, definde in source datalaoder)

--------------------------------------------------------------------------------------------
local nc = opt.nc
local loadSize   = {nc, opt.loadSize}
local sampleSize = {nc, opt.fineSize}

function pre_process(img)

    img=img*255;
    im2=img:clone()
    im2[{{3},{},{}}]=img[{{1},{},{}}]-123.68
    im2[{{2},{},{}}]=img[{{2},{},{}}]-116.779
    im2[{{1},{},{}}]=img[{{3},{},{}}]-103.939
    im2=im2/255;
    img=im2:clone()
   return img
end
local function loadImage(path)
   local input = image.load(path, nc, 'float')
   -- find the smaller dimension, and resize it to loadSize[2] (while keeping aspect ratio)
   if loadSize[2]>0 then
     local iW = input:size(3)
     local iH = input:size(2)
     if iW < iH then
        input = image.scale(input, loadSize[2], loadSize[2] * iH / iW)
     else
        input = image.scale(input, loadSize[2] * iW / iH, loadSize[2])
     end
   elseif loadSize[2]<0 then
    local scalef = 0
     if loadSize[2] == -1 then
       scalef = torch.uniform(0.5,1.5)
     else
       scalef = torch.uniform(1,3)
     end
     local iW = scalef*input:size(3)
     local iH = scalef*input:size(2)
     input = image.scale(input, iH, iW)
   end
   return input
end

-- channel-wise mean and std. Calculate or load them from disk later in the script.
local mean,std
--------------------------------------------------------------------------------
-- Hooks that are used for each image that is loaded

-- function to load the image, jitter it appropriately (random crops etc.)
local valHook = function(self, path)
   collectgarbage()
   local input = loadImage(path)
   local iW = input:size(3)
   local iH = input:size(2)

   -- do random crop
   local oW = sampleSize[2];
   local oH = sampleSize[2]
   local h1 = math.ceil(torch.uniform(1e-2, iH-oH))
   local w1 = math.ceil(torch.uniform(1e-2, iW-oW))
   local out = image.crop(input, w1, h1, w1 + oW, h1 + oH)
   assert(out:size(2) == oW)
   assert(out:size(3) == oH)
   -- do hflip with probability 0.5
   if torch.uniform() > 0.5 then out = image.hflip(out); end
 --  out:mul(2):add(-1) -- make it [0, 1] -> [-1, 1]
	out=pre_process(out)
	out:mul(255) -- make it [0, 1] -> [0, 255]
   return out
end

-------------------------------------
-- valLoader
if paths.filep(valCache) then
   print('Loading val metadata from cache')
   valLoader = torch.load(valCache)
   valLoader.sampleHookVal = valHook
   valLoader.loadSize = {nc, opt.loadSize, opt.loadSize}
   valLoader.sampleSize = {nc, sampleSize[2], sampleSize[2]}
else
   print('Creating val metadata')
     trainLoader = torch.load(trainCache)
     valLoader = dataLoader{
      paths = {paths.concat(opt.data, 'val')},
      loadSize = {nc, loadSize[2], loadSize[2]},
      sampleSize = {nc, sampleSize[2], sampleSize[2]},
      split = 100,
      verbose = true,
      forceClasses = {trainLoader.classes,trainLoader.Valtable} -- force consistent class indices between trainLoader and testLoader

   }
   torch.save(valCache, valLoader)
   print('saved metadata cache at', valCache)
   valLoader.sampleHookVal = valHook
end
collectgarbage()

-- if paths.filep(testCache) then
--    print('Loading test metadata from cache')
--    testLoader = torch.load(testCache)
--    testLoader.sampleHookTest = testHook
--  testLoader.loadSize = {nc, opt.loadSize, opt.loadSize}
--    testLoader.sampleSize = {nc, sampleSize[2], sampleSize[2]}
--  --  assert(testLoader.paths[1] == paths.concat(opt.data, 'val'),
--          -- 'cached files dont have the same path as opt.data. Remove your cached files at: '
--           --   .. testCache .. ' and rerun the program')
-- else
--    print('Creating test metadata')
--    testLoader = dataLoader{
--       paths = {paths.concat(opt.data, 'val')},
--      loadSize = {nc, loadSize[2], loadSize[2]},
--       sampleSize = {nc, sampleSize[2], sampleSize[2]},
--       split = 100,
--       verbose = true,
--       forceClasses = trainLoader.classes -- force consistent class indices between trainLoader and testLoader
--    }
--    torch.save(testCache, testLoader)
--    testLoader.sampleHookTest = testHook
-- end
-- collectgarbage()




-- do some sanity checks on valLoader
do
   local class = valLoader.imageClass
   local nClasses = #valLoader.classes
   assert(class:max() <= nClasses, "class logic has error")
   assert(class:min() >= 1, "class logic has error")
end









