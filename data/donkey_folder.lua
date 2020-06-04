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
paths.dofile('dataset.lua')

-- This file contains the data-loading logic and details.
-- It is run by each data-loader thread.
------------------------------------------
-------- COMMON CACHES and PATHS
-- Check for existence of opt.data
opt.data = os.getenv('DATA_ROOT')
if not paths.dirp(opt.data) then
    error('Did not find directory: ', opt.data)
end

-- a cache file of the training metadata (if doesnt exist, will be created)
local cache = "cache"
local cache_prefix = opt.data:gsub('/', '_')
os.execute('mkdir -p cache')
local trainCache = paths.concat(cache, cache_prefix .. '_trainCache.t7')
-- local testCache = paths.concat(cache, cache_prefix .. '_testCache.t7')

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
local trainHook = function(self, path)
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


-- testHook = function(self, path)
--    collectgarbage()
--    local input = loadImage(path)
--    local oH = sampleSize[2]
--    local oW = sampleSize[3]
--    local iW = input:size(3)
--    local iH = input:size(2)
--    local w1 = math.ceil((iW-oW)/2)
--    local h1 = math.ceil((iH-oH)/2)
--    local out = image.crop(input, w1, h1, w1+oW, h1+oH) -- center patch
--   assert(out:size(2) == oW)
--    assert(out:size(3) == oH)
--    -- do hflip with probability 0.5
--    if torch.uniform() > 0.5 then out = image.hflip(out); end
--   -- out:mul(2):add(-1) -- make it [0, 1] -> [-1, 1]
-- 	out:mul(255) -- make it [0, 1] -> [0, 255]
--    return out
-- end





--------------------------------------
-- trainLoader
if paths.filep(trainCache) then
   print('Loading train metadata from cache')
   trainLoader = torch.load(trainCache)
   trainLoader.sampleHookTrain = trainHook
   trainLoader.loadSize = {nc, opt.loadSize, opt.loadSize}
   trainLoader.sampleSize = {nc, sampleSize[2], sampleSize[2]}
else
   print('Creating train metadata')
   local class_odering={}
  table.insert(class_odering, 'letter_tray');
  table.insert(class_odering, 'paper_notebook');
  table.insert(class_odering, 'printer');
  table.insert(class_odering, 'bike_helmet');
  table.insert(class_odering, 'desk_lamp');
  table.insert(class_odering, 'mobile_phone');
  table.insert(class_odering, 'desk_chair');
  table.insert(class_odering, 'pen');
  table.insert(class_odering, 'phone');
  table.insert(class_odering, 'headphones');
  table.insert(class_odering, 'ring_binder');
  table.insert(class_odering, 'tape_dispenser');
  table.insert(class_odering, 'bookcase');
  table.insert(class_odering, 'back_pack');
  table.insert(class_odering, 'laptop_computer');
  table.insert(class_odering, 'stapler');
  table.insert(class_odering, 'ruler');
  table.insert(class_odering, 'mouse');
  table.insert(class_odering, 'projector');
  table.insert(class_odering, 'trash_can');
  table.insert(class_odering, 'monitor');
  table.insert(class_odering, 'file_cabinet');
  table.insert(class_odering, 'speaker');
  table.insert(class_odering, 'punchers');
  table.insert(class_odering, 'desktop_computer');
  table.insert(class_odering, 'bottle');
  table.insert(class_odering, 'mug');
  table.insert(class_odering, 'keyboard');
  table.insert(class_odering, 'scissors');
  table.insert(class_odering, 'bike');
  table.insert(class_odering, 'calculator');

  local Valtable={}
  for ii=1,opt.number_of_testclass do
    print('class_odering[ii]',class_odering[ii])
    table.insert(Valtable, class_odering[ii]);
  end

  -- Valtable={'letter_tray','paper_notebook'} -- put class in order


   trainLoader = dataLoader{
      paths = {paths.concat(opt.data, 'train')},
      loadSize = {nc, loadSize[2], loadSize[2]},
      sampleSize = {nc, sampleSize[2], sampleSize[2]},
      split = 100,
      verbose = true,
      forceClasses = {class_odering,Valtable} -- force consistent class indices between trainLoader and testLoader
   }
   torch.save(trainCache, trainLoader)
   print('saved metadata cache at', trainCache)
   trainLoader.sampleHookTrain = trainHook
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




-- do some sanity checks on trainLoader
do
   local class = trainLoader.imageClass
   local nClasses = #trainLoader.classes
   assert(class:max() <= nClasses, "class logic has error")
   assert(class:min() >= 1, "class logic has error")
end









