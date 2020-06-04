--------------------------------------------------------
-- Torch Implementation of  Attending to Discriminative Certainty for Domain Adaptation(CVPR2019)
--- Written by  jointly Vinod Kumar Kurmi (vinodkumarkurmi@gmail.com) and Shanu Kumar
require 'cutorch'
require 'nngraph'
print("pass")
require 'math'
require 'cunn'
require 'optim'
require 'gnuplot'
require 'loadcaffe'
require 'image';
require 'torch';
require 'nn';
require 'xlua'
require 'loadcaffe'
require 'cudnn'
require 'nnlr' --- for layer wise learnig rate

require 'distributions';
bayesian_crossentropy = require 'bayesian_crossentropy';
require 'cephes'
----------------------------------------------------
local c = require 'trepl.colorize'
local data_tm = torch.Timer()
----------------------------------------------------
opt = {
    manual_seed=1,          -- Seed
	batchSize = 20,         -- batch Size
	Test_batchSize = 40,     -- Test time batch size, it may change after the last epoch of test data
	start_Batch_IndexTest=1, --batch index at time of testing
	loadSize = 256,         -- resize the loaded image to loadsize maintaining aspect ratio. -- see donkey_folder.lua
	fineSize = 224,         -- size of random crops
	nc = 3,                 -- # of channels in input
	nThreads = 1,           -- #  of data loading threads to use
	gpu = 1,                -- gpu = 0 is CPU mode. gpu=X is GPU mode on GPU X
	save='logs/',            -- Saving the logs of trainining
	net1_freeze='yes',      -- For not updating the first 3 Conv layer
	--momentum
	number_of_testclass=31,  -- number of class in test dataset, in general it is =source class but we can use less class also.
    lamda=1,                -- Lamda value for gradeint reversal value.(fix)
	momentum=0.9,
	baseLearningRate=0.0002,
	max_epoch=6000,
	gamma=0.001,   -- for inverse policy :  base_lr * (1 + gamma * iter) ^ (- power)
	power=0.75,    -- for inverse policy :  base_lr * (1 + gamma * iter) ^ (- power)
	max_epoch_grl=10000, -- For progress in process , calculate the lamda for grl
	alpha=10,  -- LR schdular (2nd way)
	num_sim = 100, -- monte_carlo_simulations
	alpha_0_cls = 100,
	alpha_0_dis = 15,
	al_wt_for_dis=100,
	al_wt_for_clw=100,
}
	cutorch.manualSeed(opt.manual_seed)
	torch.manualSeed(opt.manual_seed)



--=====================Tuning Parameters===================================
	local prev_accuracy = 0
	local max_accuracy = 0
	batchSize =opt.batchSize
	opt.save=opt.save .. 'aleatoric_batchsize_' .. opt.batchSize
	torch.setnumthreads(1)
	torch.setdefaulttensortype('torch.FloatTensor')
--==============Ploting Fuction=============================================================================
	confusion = optim.ConfusionMatrix({'letter_tray','paper_notebook','printer','bike_helmet','desk_lamp','mobile_phone',
		'desk_chair','pen','phone','headphones','ring_binder','tape_dispenser','bookcase','back_pack','laptop_computer','stapler',
		'ruler','mouse','projector','trash_can','monitor','file_cabinet','speaker','punchers','desktop_computer','bottle',
		'mug','keyboard','scissors','bike','calculator'})

	print('Will save at '..opt.save)
	paths.mkdir(opt.save)
-- 	testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))
-- 	testLogger:setNames{'% mean class accuracy (train set)', '% mean class accuracy (test set)'}
-- 	testLogger.showPlot = false
	-------------logger for accuracy------------
	acc_Logger = optim.Logger(paths.concat(opt.save, 'acc.log'))
	acc_Logger:setNames{'Train acc', 'Test acc'}
	acc_Logger.showPlot = false
	----------------logger for loss----------------
	error_Logger = optim.Logger(paths.concat(opt.save, 'error.log'))
	error_Logger:setNames{'Train error', 'Test error', 'Train error cross', 'Test error cross'}
	error_Logger.showPlot = false
	---------------logger for entropy-----------------
	entropy_Logger = optim.Logger(paths.concat(opt.save, 'entropy.log'))
	entropy_Logger:setNames{'Train source entropy', 'Train target entropy', 'Test entropy'}
	entropy_Logger.showPlot = false
	---------------logger for differential entropy-----------------
	entropy_diff_Logger = optim.Logger(paths.concat(opt.save, 'diff_entropy.log'))
	entropy_diff_Logger:setNames{'Train source diff entropy', 'Train target diff entropy', 'Test diff entropy'}
	entropy_diff_Logger.showPlot = false
	---------------logger for var-----------------
	var_Logger = optim.Logger(paths.concat(opt.save, 'var.log'))
	var_Logger:setNames{'Train source var', 'Train target var', 'Test var'}
	var_Logger.showPlot = false
	----------------logger for brier score----------------
	brier_Logger = optim.Logger(paths.concat(opt.save, 'brier.log'))
	brier_Logger:setNames{'Train brier', 'Test brier'}
	brier_Logger.showPlot = false
    ----------------logger for domain aleatoric loss----------------
	error_domain_aleatoric_Logger = optim.Logger(paths.concat(opt.save, 'error_domain_aleatoric.log'))
	error_domain_aleatoric_Logger:setNames{'Train source aleatoric error', 'Train target aleatoric error', 'Test target aleatoric error'}
	error_domain_aleatoric_Logger.showPlot = false
    ----------------logger for domain aleatoric uncertainty----------------
	aleatoric_domain_uncertainty_Logger = optim.Logger(paths.concat(opt.save, 'aleatoric_domain_uncertainty.log'))
	aleatoric_domain_uncertainty_Logger:setNames{'Train source aleatoric domain uncertainty', 'Train target aleatoric domain uncertainty', 'Test target aleatoric domain uncertainty'}
	aleatoric_domain_uncertainty_Logger.showPlot = false
    ----------------logger for classification aleatoric loss----------------
	error_aleatoric_Logger = optim.Logger(paths.concat(opt.save, 'error_aleatoric.log'))
	error_aleatoric_Logger:setNames{'Train source aleatoric error', 'Test target aleatoric error'}
	error_aleatoric_Logger.showPlot = false
    ----------------logger for classification aleatoric uncertainty----------------
	aleatoric_uncertainty_Logger = optim.Logger(paths.concat(opt.save, 'aleatoric_uncertainty.log'))
	aleatoric_uncertainty_Logger:setNames{'Train source aleatoric uncertainty', 'Train target aleatoric uncertainty', 'Test target aleatoric uncertainty'}
	aleatoric_uncertainty_Logger.showPlot = false
--==========================================================================================-------------------     -------------------------Path Initilication ----------------
	 net_orignal=torch.load('../../../../../Pretrained_models/pretrained_netwrok_resnet/resnet-50.t7')
	-- prototxt_name = '../pretrained_network/deploy.prototxt'
	-- binary_name = '../pretrained_network/bvlc_alexnet.caffemodel'
	-- net_orignal = loadcaffe.load(prototxt_name, binary_name,'cudnn');
	-- print(' net_orignal', net_orignal)

    ---------------------------------------------------------
	-- create Train data loader
	local DataLoader = paths.dofile('data/data.lua')
	local data = DataLoader.new(opt.nThreads, opt)
	print("Train Dataset Size: ", data:size())

	-- create Val data loader
	local DataLoaderVal = paths.dofile('data/data_target.lua')
	local dataVal = DataLoaderVal.new(opt.nThreads, opt)
	print("Val Dataset Size: ", dataVal:size())

	-- create Test data loader
	local DataLoaderTest = paths.dofile('data_test/data.lua')
	local dataTest = DataLoaderTest.new(0, opt)
	print("test new Dataset Size: ", dataTest:size())

    ----------------------------------------------------
	--===FUNCTIONS==============

	function uti(filename)
	   local net = torch.load(filename)
	  net:apply(function(m) if m.weight then
		 m.gradWeight = m.weight:clone():zero();
		 m.gradBias = m.bias:clone():zero(); end end)
	   return net
	end

	function lgamma(input)
		local temp = (cephes.lgam(input:float())):type(input:type())
		return temp:resizeAs(input)
	end

	function digamma(input)
		local temp = (cephes.digamma(input:float())):type(input:type())
		return temp:resizeAs(input)
	end

	function check_accuracy(scores, targets)
		local num_test = (#targets)[1]
		local no_correct = 0
		local confidences, indices = torch.sort(scores, true)
		local predicted_classes = indices[{{},{1}}]:long()
		targets = targets:long()
		no_correct = no_correct + ((torch.squeeze(predicted_classes):eq(targets)):sum())
		local accuracy = no_correct / num_test
		return accuracy
	end


	function check_accuracyTest(scores, targets)
        local num_test = (#targets)[1]
        local no_correct = 0
        local confidences, indices = torch.sort(scores, true)
        local predicted_classes = indices[{{},{1}}]:long()
        targets = targets:long()
        if num_test==1 then
            no_correct = no_correct + ((predicted_classes:eq(targets)):sum())
        else
            no_correct = no_correct + ((torch.squeeze(predicted_classes):eq(targets)):sum())
        end
        local accuracy = no_correct
        return accuracy
    end

	function round(what, precision)
		return math.floor(what*math.pow(10,precision)+0.5) / math.pow(10,precision)
	end

	function TableConcat(t1,t2)
		for i=1,#t2 do
			t1[#t1+1] = t2[i]
		end
		return t1
	end
--===============================feature extractor======================================================
	function build_feature_extractor()
        net1= nn.MapTable()
        net2= nn.MapTable()
        net3= nn.MapTable()
        netB= nn.MapTable()

        net11= nn.Sequential()
        net22= nn.Sequential()
        net33= nn.Sequential()
        net88= nn.Sequential()


        netBB= nn.Sequential()

    -- Layer by layer copy from the pretrained Alexnet  Netwrok
    for i, module in ipairs(net_orignal.modules) do
         if(i<5) then
            net11:add(module):learningRate('weight', 1)
                            :learningRate('bias', 2)
                            :weightDecay('weight', 1)
                            :weightDecay('bias', 0) --conv4


        elseif (i==5) then
                net11:add(module):learningRate('weight', 1)
                          :learningRate('bias', 2)
                          :weightDecay('weight', 1)
                          :weightDecay('bias', 0) --conv4

        elseif (i==6) then
                net11:add(module):learningRate('weight', 1)
                          :learningRate('bias', 2)
                          :weightDecay('weight', 1)
                          :weightDecay('bias', 0) --FC6
        elseif (i==7) then
            net22:add(module):learningRate('weight', 1)
                        :learningRate('bias', 2)
                        :weightDecay('weight', 1)
                        :weightDecay('bias', 0) --FC6


        elseif (i==8) then
            net22:add(module):learningRate('weight', 1)
                        :learningRate('bias', 2)
                        :weightDecay('weight', 1)
                        :weightDecay('bias', 0) --FC6


        elseif (i>8 and i<11 ) then
            net33:add(module):learningRate('weight', 1)
                        :learningRate('bias', 2)
                        :weightDecay('weight', 1)
                        :weightDecay('bias', 0) --FC6

        else

            net88:add(module):learningRate('weight', 1)
                        :learningRate('bias', 2)
                        :weightDecay('weight', 1)
                        :weightDecay('bias', 0) --FC6

         end
    end

        net1:add(net11)
        net2:add(net22)
        net3:add(net33)
      -- Bottlenec Network------
    netBB:add(nn.Linear( 2048, 256)):learningRate('weight', 10)
                          :learningRate('bias', 20)
                          :weightDecay('weight', 1)
                          :weightDecay('bias', 0)
    netBB:add(nn.ReLU(true))

    netB:add(netBB)
    return net1,net2,net3,netB



	end

	function build_classifier()
		-- Classifier Network------------------
		local net4 = nn.MapTable()
		local net44 = nn.Sequential()
		net44:add( nn.Linear(256, 128)):learningRate('weight', 10)
							  :learningRate('bias', 20)
		net44:add(nn.ReLU(true))
		------stochastic inference----------
		net44:add(nn.Dropout(0.5, nil, nil, true))
-- 		net44:add(nn.LogSoftMax())
		net4:add(net44)
		return net4
	end

    function build_classifier_logit()
        local net5 = nn.MapTable()
        local net55 = nn.Sequential()
        net55:add( nn.Linear(128, 31)):learningRate('weight', 10)
						  :learningRate('bias', 20)
						  :weightDecay('weight', 1)
						  :weightDecay('bias', 0) --FC6
        net5:add(net55)
        return net5
    end

    function build_classifier_variance()
        local netv = nn.MapTable()
        local netvv = nn.Sequential()
        netvv:add(nn.Linear(128, 1)):learningRate('weight', 10)
							  :learningRate('bias', 20)
                            :weightDecay('weight', 1)
						  :weightDecay('bias', 0) --FC6
        netvv:add(nn.SoftPlus())
        netv:add(netvv)
        return netv
    end

	function build_discriminator()
		local netD = nn.MapTable()
		local netDD = nn.Sequential()
		module = nn.GradientReversal(lambda)
		netDD:add(module)
		netDD:add( nn.Linear( 256, 1024)):learningRate('weight', 10)
							  :learningRate('bias', 20)
		netDD:add(nn.ReLU(true))
		netDD:add(nn.Dropout(0.5, nil, nil, false))
		netDD:add( nn.Linear( 1024, 1024)):learningRate('weight', 10)
							  :learningRate('bias', 20)
		netDD:add(nn.ReLU(true))
		netDD:add(nn.Dropout(0.5, nil, nil, false))
-- 		netDD:add( nn.Linear( 1024, 2)):learningRate('weight', 10)
-- 							  :learningRate('bias', 20)
		--netDD:add(nn.Sigmoid())  -- removed this layer if we are using the nn.CrossEntropyCriterion()
		netD:add(netDD)
		 --Initially Lamda set =0
		module:setLambda(0)
		return netD
	end

	function build_discriminator_logit()
		local netDL = nn.MapTable()
		local netDDL = nn.Sequential()
		netDDL:add(nn.Linear(1024, 2)):learningRate('weight', 10)
							  :learningRate('bias', 20)
		netDL:add(netDDL)
		return netDL
	end
	function build_discriminator_var()
		local netDV = nn.MapTable()
		local netDDV = nn.Sequential()
		netDDV:add(nn.Linear(1024, 1)):learningRate('weight', 10)
							  :learningRate('bias', 20)
		netDDV:add(nn.SoftPlus())
		netDV:add(netDDV)
		return netDV
	end
-- ================= Models ====================

	net4 = build_classifier()
    net5 = build_classifier_logit()  -- compute logits
    netV = build_classifier_variance() -- compute variance
	net1, net2, net3, netB = build_feature_extractor()
	netD = build_discriminator()
	netDL = build_discriminator_logit()
	netDV = build_discriminator_var()
	softmax = nn.MapTable()
    softmax:add(nn.SoftMax())
    logsoftmax = nn.MapTable()
    logsoftmax:add(nn.LogSoftMax())
	onehot = nn.MapTable()
	onehot:add(nn.OneHot(31))
	onehot_dis = nn.MapTable()
	onehot_dis:add(nn.OneHot(2))--============ Criterion=================
	local criterion = nn.CrossEntropyCriterion()
	local criterionNLL = nn.ClassNLLCriterion()
	local criterionNLL_parallel = nn.ParallelCriterion():add(criterionNLL):add(criterionNLL)
	local criterionCrossE = nn.CrossEntropyCriterion()
	local criterionCrossE_parallel = nn.ParallelCriterion():add(criterionCrossE):add(criterionCrossE)
    -- local criterion_bayesian = autograd.nn.AutoCriterion('AutoMSE')(bayesian_categorical_crossentropy)
    local criterion_bayesian_class = bayesian_crossentropy.criterion(opt.num_sim, 31)
    local criterion_bayesian_dis = bayesian_crossentropy.criterion(opt.num_sim, 2)

--==========================================

-----------------------------------------------------------------------------------------------
	if opt.gpu >=0 then
		net1:cuda()
		net2:cuda()
		net3:cuda()
		netB:cuda()
		net4:cuda()
        net5:cuda()
        netV:cuda()
		netD:cuda()
		netDL:cuda()
		netDV:cuda()
		criterion:cuda()
		criterionNLL_parallel:cuda()
		criterionCrossE_parallel:cuda()
		softmax:cuda()
		onehot:cuda()
		onehot_dis:cuda()
		logsoftmax:cuda()
        criterion_bayesian_dis:cuda()
		criterion_bayesian_class:cuda()
	 end

--=== Different Learning rate for weigth and bias
	local temp_baseWeightDecay=0.001  --no meaningin my case
	local learningRates_Net1, weightDecays_Net1 = net1:getOptimConfig(opt.baseLearningRate,temp_baseWeightDecay)
	local learningRates_Net2, weightDecays_Net2 = net2:getOptimConfig(opt.baseLearningRate, temp_baseWeightDecay)
	local learningRates_Net3, weightDecays_Net3 = net3:getOptimConfig(opt.baseLearningRate, temp_baseWeightDecay)
	local learningRates_NetB, weightDecays_NetB = netB:getOptimConfig(opt.baseLearningRate, temp_baseWeightDecay)
	local learningRates_NetD, weightDecays_NetD = netD:getOptimConfig(opt.baseLearningRate, temp_baseWeightDecay)
	local learningRates_Net4, weightDecays_Net4 = net4:getOptimConfig(opt.baseLearningRate, temp_baseWeightDecay)
    local learningRates_Net5, weightDecays_Net5 = net5:getOptimConfig(opt.baseLearningRate, temp_baseWeightDecay)
	local learningRates_NetDL, weightDecays_NetDL = netDL:getOptimConfig(opt.baseLearningRate, temp_baseWeightDecay)
    local learningRates_NetDV, weightDecays_NetDV = netDV:getOptimConfig(opt.baseLearningRate, temp_baseWeightDecay)
	local learningRates_NetV, weightDecays_NetV = netV:getOptimConfig(opt.baseLearningRate, temp_baseWeightDecay)

		--===========Parameters===================================
	parameters1, gradParameters1 = net1:getParameters()
	parameters2, gradParameters2 = net2:getParameters()
	parameters3, gradParameters3 = net3:getParameters()
	parametersB, gradParametersB = netB:getParameters()
	parametersD, gradParametersD = netD:getParameters()
	parameters4, gradParameters4 = net4:getParameters()
    parameters5, gradParameters5 = net5:getParameters()
	parametersDL, gradParametersDL = netDL:getParameters()
    parametersDV, gradParametersDV = netDV:getParameters()
    parametersV, gradParametersV = netV:getParameters()
--============================================================
	local method = 'xavier'
	net4 = require('misc/weight-init')(net4, method)
    net5 = require('misc/weight-init')(net5, method)
    netV = require('misc/weight-init')(netV, method)
	netB = require('misc/weight-init')(netB, method)
	netD = require('misc/weight-init')(netD, method)
	netDL = require('misc/weight-init')(netDL, method)
	netDV = require('misc/weight-init')(netDV, method)
----------------------------------------------------------------------------------------------------
	print('=> New Model')
	print(model)
	print('net1', net1)
	print('net2', net2)
	print('net3', net3)
	print('netB', netB)
	print("netD", netD)
	print("net4", net4)
    print("net5", net5)
    print("netV", netV)
	print("netDL", netDL)
    print("netDV", netDV)
	print(criterion)
	collectgarbage()
	local updated_learningrate=opt.baseLearningRate

--===================Training Fuctions======================================

function train()
	net1:training()
	net2:training()
	net3:training()
	net4:training()
    net5:training()
    netV:training()
	netB:training()
	netD:training()
	netDL:training()
	netDV:training()
	epoch = epoch or 1
	if(epoch>1) then
	print(c.blue '==>'.." online epoch # " .. epoch .. ' [batchSize = ' .. batchSize .. ']')
		local p=epoch/opt.max_epoch_grl
		local baseWeightDecay = torch.pow((1 +  epoch * opt.gamma), (-1  * opt.power)) -- need to chanage
            updated_learningrate=opt.baseLearningRate*baseWeightDecay
		print('Learnig Rate',updated_learningrate)
		--opt.lamda=(2*torch.pow(1+torch.exp(-10*p),-1))-1
		print('Lamda', opt.lamda)
		module:setLambda(opt.lamda)
	end
	local avg_loss = 0
	local avg_loss_cross = 0
	local avg_aleatoric_loss_sr = 0                   --- for source
	local avg_aleatoric_domain_loss_sr = 0            --- for source
	local avg_aleatoric_domain_loss_tg = 0            --- for target
    local avg_aleatoric_domain_uncertainty_sr = 0     --- for source
	local avg_aleatoric_domain_uncertainty_tg = 0     --- for target
	local avg_aleatoric_uncertainty_sr = 0
	local avg_aleatoric_uncertainty_tg = 0
	local avg_acc = 0
	local avg_entropy_sr = 0
	local avg_entropy_tg = 0
	local avg_brier = 0
	local avg_diff_entropy_sr = 0
	local avg_diff_entropy_tg = 0
	local avg_dir_var_sr = 0
	local avg_dir_var_tg = 0
	local count = 0
	for i = 1, data:size(), opt.batchSize do
		-----Classifier Network-------------------
		data_tm:reset(); data_tm:resume()
		local batchInputs_source,label = data:getBatch()
		local batchInputs_target = dataVal:getBatch()
		local SlabelDomain = torch.Tensor(opt.batchSize):fill(1)
		local TlabelDomain = torch.Tensor(opt.batchSize):fill(2)
		local smooth_factor_cls = torch.Tensor(opt.batchSize, 31):fill((opt.alpha_0_cls - 31)/opt.alpha_0_cls)
		local bias_cls = torch.Tensor(opt.batchSize, 31):fill(1/opt.alpha_0_cls)
-- 		local smooth_factor_dis = torch.Tensor(opt.batchSize, 2):fill((opt.alpha_0_dis - 31)/opt.alpha_0_dis)
-- 		local bias_dis = torch.Tensor(opt.batchSize, 2):fill(1/opt.alpha_0_dis)
		if opt.gpu >=0 then
			label=label:cuda()
			SlabelDomain = SlabelDomain:cuda()
			TlabelDomain = TlabelDomain:cuda()
			batchInputs_source = batchInputs_source:cuda()
			batchInputs_target = batchInputs_target:cuda()
			smooth_factor_cls = smooth_factor_cls:cuda()
			bias_cls = bias_cls:cuda()
-- 			smooth_factor_dis = smooth_factor_dis:cuda()
-- 			bias_dis = bias_dis:cuda()
		end
        -------- forwardNetwork -------------------------
		outputs1 = net1:forward({batchInputs_source,batchInputs_target})
		outputs2 = net2:forward(outputs1)
		outputs3 = net3:forward(outputs2)
		outputsB = netB:forward(outputs3)
		-- outputs4 = net4:forward(outputsB)
        -- outputs5 = net5:forward(outputs4)  -- predicted class logits
        -- outputsV = netV:forward(outputs4)  -- predicted variance
		outputsD = netD:forward(outputsB)
        outputsDL = netDL:forward(outputsD)  -- predicted domain logits
		outputsDV = netDV:forward(outputsD)  -- predicted domain variance

        --------------------- backward Network ------------------------------------------
		gradParametersD:zero()
		gradParametersDL:zero()
		gradParametersDV:zero()
		gradParameters4:zero()
        gradParameters5:zero()
        gradParametersV:zero()
		gradParametersB:zero()
		gradParameters3:zero()
		gradParameters2:zero()
		gradParameters1:zero()


		---Aleatoric local attenstion from discriminator using uncertainty (certainty)
		local outputsB_source = outputsB[1]:clone()
		local outputsB_target = outputsB[2]:clone()
		gradParametersD:zero()
		local outputsB_source1 = outputsB[1]:clone()
		local outputsB_target1 = outputsB[2]:clone()
		local dgradOutputs_modDV_aletoric  = netDV:backward(outputsD, outputsDV)
		local dgradOutputs_modD_aletoric  = netD:backward({outputsB_source1, outputsB_target1}, dgradOutputs_modDV_aletoric)

		 outputsDV_clone1= outputsDV[1]:clone()
		 outputsDV_clone2= outputsDV[2]:clone()
		local activations_source =outputsB[1]:clone()
		local activations_target =outputsB[2]:clone()
		activations_source:squeeze()
		activations_target:squeeze()
		local gradients_source = dgradOutputs_modD_aletoric[1]:squeeze()
		local gradients_target = dgradOutputs_modD_aletoric[2]:squeeze()

		local map_source = torch.cmul(activations_source, gradients_source)
		 map_source = map_source:cmul(torch.gt(map_source,0):typeAs(map_source))
		 map_source[map_source:le(0)] = -1000
		 local soft_source=nn.SoftMax():cuda():forward(map_source)
		local one_tensor = torch.CudaTensor(outputsDV_clone1:size()):fill(1)
		local weight_DV1 = torch.csub(one_tensor, outputsDV_clone1)
		local soft_after_mul_source = soft_source:cmul(weight_DV1:view(opt.batchSize,1):expand(opt.batchSize,256))
		local soft_clamp_source = torch.clamp(soft_after_mul_source, 0, 1)
        attentive_weight_source=torch.CudaTensor()    --Declaration of dgradOutputsS for source class
		attentive_weight_source:resize(outputsB[1]:size())
		attentive_weight_source:fill(1)
		attentive_weight_source = attentive_weight_source + soft_clamp_source
		effective_feature_source = torch.cmul(outputsB_source, attentive_weight_source)

		local map_target = torch.cmul(activations_target, gradients_target)
		 map_target = map_target:cmul(torch.gt(map_target,0):typeAs(map_target))
		 map_target[map_target:le(0)] = -1000
		local soft_target=nn.SoftMax():cuda():forward(map_target)
		local one_tensor = torch.CudaTensor(outputsDV_clone2:size()):fill(1)
		local weight_DV2 = torch.csub(one_tensor, outputsDV_clone2)
		local soft_after_mul_target = soft_target:cmul(weight_DV2:view(opt.batchSize,1):expand(opt.batchSize,256))
		local soft_clamp_target = torch.clamp(soft_after_mul_target, 0, 1)
		attentive_weight_target=torch.CudaTensor()    --Declaration of dgradOutputsS for source class
		attentive_weight_target:resize(outputsB[2]:size())
		attentive_weight_target:fill(1)
		attentive_weight_target = attentive_weight_target + soft_clamp_target
		effective_feature_target = torch.cmul(outputsB_target, attentive_weight_target)
		--------------------------------------------------------------------------------------------------------------------
		gradParametersD:zero()
		gradParametersDL:zero()
		gradParametersDV:zero()
		gradParameters4:zero()
        gradParameters5:zero()
        gradParametersV:zero()
		gradParametersB:zero()
		gradParameters3:zero()
		gradParameters2:zero()
		gradParameters1:zero()

		outputsD = netD:forward(outputsB)
        outputsDL = netDL:forward(outputsD)  -- predicted domain logits
		outputsDV = netDV:forward(outputsD)  -- predicted domain variance

        -------- domain cross entropy ----------
		errDomain = criterionCrossE_parallel:forward(outputsDL, {SlabelDomain,TlabelDomain})
		--- domain classification loss grad -------------
		dgradOutputsDomain_Classifier = criterionCrossE_parallel:backward(outputsDL, {SlabelDomain,TlabelDomain})

		------- Gradient for aleatoric loss (domain classification + variance) -----------
		local dgradOutputsDomain_Bayes = {}
		------ aleatoric domain loss for source -------------------
        err_aleatoric_dom_source = criterion_bayesian_dis:forward({outputsDL[1], SlabelDomain, outputsDV[1], torch.CudaTensor(opt.batchSize):fill(1), torch.CudaTensor(1):fill(opt.num_sim)})[1]
		---- gradient for aleatoric loss (source domain batch) ------
		dgradOutputsDomain_Bayes[1] = criterion_bayesian_dis:backward({outputsDL[1], SlabelDomain, outputsDV[1], torch.CudaTensor(opt.batchSize):fill(1), torch.CudaTensor(1):fill(opt.num_sim)}, torch.CudaTensor(1):fill(0.2))

		------ aleatoric domain loss for target -------------------
		err_aleatoric_dom_target = criterion_bayesian_dis:forward({outputsDL[2], TlabelDomain, outputsDV[2], torch.CudaTensor(opt.batchSize):fill(1), torch.CudaTensor(1):fill(opt.num_sim)})[1]
		---- gradient for aleatoric loss (target domain batch) ------
		dgradOutputsDomain_Bayes[2] = criterion_bayesian_dis:backward({outputsDL[2], TlabelDomain, outputsDV[2], torch.CudaTensor(opt.batchSize):fill(1), torch.CudaTensor(1):fill(opt.num_sim)}, torch.CudaTensor(1):fill(0.2))

		---- gradient for domain classifier (gradient domain classification + gradient domain variance)
		dgradOutputsDomain = {torch.add(dgradOutputsDomain_Classifier[1], dgradOutputsDomain_Bayes[1][1]), torch.add(dgradOutputsDomain_Classifier[2], dgradOutputsDomain_Bayes[2][1])}

        ---- Optimization Domain Confusion Branch (netDL) -------
		feval_netDL = function(x)
			dgradOutputs_modDL  = netDL:backward(outputsD, dgradOutputsDomain)
			return err, gradParametersDL
		end
		optim.sgd(feval_netDL, parametersDL, {
								   learningRates = learningRates_NetDL,
								   weightDecays = weightDecays_NetDL,
								   learningRate = updated_learningrate,
								   momentum = opt.momentum,
								  })

		-----Optimzation NetDV --------
        dgradOutputsDV = {dgradOutputsDomain_Bayes[1][3], dgradOutputsDomain_Bayes[2][3]}
        feval_netDV = function(x)
		dgradOutputs_modDV = netDV:backward(outputsD, dgradOutputsDV)
			return err, gradParametersDV
		end
		optim.sgd(feval_netDV, parametersDV, {
								   learningRates = learningRates_NetDV,
								   weightDecays = weightDecays_NetDV,
								   learningRate = updated_learningrate,
								   momentum = opt.momentum,
								  })
		---- Optimization NetD -------
        local total_gradD = {}
		total_gradD[1] = dgradOutputs_modDL[1] + dgradOutputs_modDV[1]
		total_gradD[2] = dgradOutputs_modDL[2] + dgradOutputs_modDV[2]

        feval_netD = function(x)
		dgradOutputs_modD = netD:backward(outputsB, total_gradD)
			return err, gradParametersD
		end
		optim.sgd(feval_netD, parametersD, {
								   learningRates = learningRates_NetD,
								   weightDecays = weightDecays_NetD,
								   learningRate = updated_learningrate,
								   momentum = opt.momentum,
								  })

		outputs4 = net4:forward({effective_feature_source, effective_feature_target})
        outputs5 = net5:forward(outputs4)  -- predicted class logits
        outputsV = netV:forward(outputs4)  -- predicted variance

        err_aleatoric_cl_sr = criterion_bayesian_class:forward({outputs5[1], label, outputsV[1], torch.CudaTensor(opt.batchSize):fill(1), torch.CudaTensor(1):fill(opt.num_sim)})[1]
		 -------------Gradient for classification -----------
		local dgradOutputsS = torch.CudaTensor()    --Declaration of dgradOutputsS for source class
		dgradOutputsS:resize(outputs5[1]:size())
		dgradOutputsS:zero()

		------------ cross entropy for classification ------------------------------
		--- smooth the true labels as  its diffcult to match 1-hot continous distributions
		local smooth_label_cls = torch.add(torch.cmul(onehot:forward({label})[1], smooth_factor_cls), bias_cls)

		--- all the labels are possible as labels are smoothed
		local all_label_cls = torch.CudaTensor(31)
		for c = 1, 31 do
			all_label_cls[c] = c
		end

		err = 0
		for b = 1, opt.batchSize do
			local criterion_weighted = nn.CrossEntropyCriterion(smooth_label_cls[b])
			criterion_weighted:cuda()
			local temp_logits = outputs5[1][b]:view(1, outputs5[1][b]:size()[1])
			err = err + criterion_weighted:forward(torch.expand(temp_logits, temp_logits:size()[2], 31) , all_label_cls)
			dgradOutputsS[b] = torch.sum(criterion_weighted:backward(torch.expand(temp_logits, temp_logits:size()[2], 31) , all_label_cls), 1)
			criterion_weighted = nil
		end

        ------- Gradient for aleatoric loss (classification + variance) -----------
		dgradOutputsBV = criterion_bayesian_class:backward({outputs5[1], label, outputsV[1], torch.CudaTensor(opt.batchSize):fill(1), torch.CudaTensor(1):fill(opt.num_sim)}, torch.CudaTensor(1):fill(0.2))

        -- Zero gradient for Target data Classification(we dont have target label)
        local zeros = torch.CudaTensor()
		zeros:resize(dgradOutputsS:size())
		zeros:zero()
		dgradOutputs = {torch.add(dgradOutputsS, dgradOutputsBV[1]), zeros}
-- 		dgradOutputs = {dgradOutputsS, zeros}

-- 		dgradOutputs = {dgradOutputsS, zeros}

		---- Optimization Net5 -------
		feval_net5 = function(x)
		dgradOutputs_mod5 = net5:backward(outputs4, dgradOutputs)
			return err, gradParameters5
		end
		optim.sgd(feval_net5, parameters5, {
								   learningRates = learningRates_Net5,
								   weightDecays = weightDecays_Net5,
								   learningRate = updated_learningrate,
								   momentum = opt.momentum,
								  })

		-----Optimzation NetV --------
        dgradOutputsV = dgradOutputsBV[3]    -- gradient for variance of Source data
        local zerov = torch.CudaTensor()     -- Zero gradient for Target data Variance
		zerov:resize(dgradOutputsV:size())
		zerov:zero()
        dgradOutputV = {dgradOutputsV, zerov}

        feval_netV = function(x)
		dgradOutputs_modV = netV:backward(outputs4, dgradOutputV)
			return err, gradParametersV
		end
		optim.sgd(feval_netV, parametersV, {
								   learningRates = learningRates_NetV,
								   weightDecays = weightDecays_NetV,
								   learningRate = updated_learningrate,
								   momentum = opt.momentum,
								  })



    ---- Optimization Net4 -------
		net4:zeroGradParameters()
		gradParameters4:zero()
        local total_grad4 = {}
		total_grad4[1] = dgradOutputs_mod5[1] + dgradOutputs_modV[1]
		total_grad4[2] = dgradOutputs_mod5[2] + dgradOutputs_modV[2]

        feval_net4 = function(x)
		dgradOutputs_mod4 = net4:backward({effective_feature_source, effective_feature_target}, total_grad4)
			return err, gradParameters4
		end
		optim.sgd(feval_net4, parameters4, {
								   learningRates = learningRates_Net4,
								   weightDecays = weightDecays_Net4,
								   learningRate = updated_learningrate,
								   momentum = opt.momentum,
								  })

        ---- Optimization netB (bottleneck_Branch)-------
		local total_grad={}
		total_grad[1] = torch.cdiv(dgradOutputs_mod4[1], attentive_weight_source) + dgradOutputs_modD[1]
		total_grad[2] = torch.cdiv(dgradOutputs_mod4[2], attentive_weight_target) + dgradOutputs_modD[2]
		feval_netB = function(x)
			dgradOutputs_modB   = netB:backward(outputs3, total_grad)
			return err, gradParametersB
		end
		optim.sgd(feval_netB, parametersB, {
								   learningRates = learningRates_NetB,
								   weightDecays = weightDecays_NetB,
								   learningRate = updated_learningrate,
								   momentum = opt.momentum,
								  })

        ---- Optimization net3(FC6,FC7) Branch -------
		gradParameters3:zero()
		feval_net3 = function(x)
		dgradOutputs_mod3= net3:backward(outputs2,dgradOutputs_modB)
			return  err, gradParameters3
		end
		optim.sgd(feval_net3, parameters3, {
								   learningRates = learningRates_Net3,
								   weightDecays = weightDecays_Net3,
								   learningRate = updated_learningrate,
								   momentum = opt.momentum,
								  })
        ---- Optimization net2(Conv4 -Pool5) Branch -------
		gradParameters2:zero()
			feval_net2 = function(x)
		dgradOutputs_mod2= net2:backward(outputs1,dgradOutputs_mod3)
			return  err, gradParameters2
		end
		optim.sgd(feval_net2, parameters2, {
								   learningRates = learningRates_Net2,
								   weightDecays = weightDecays_Net2,
								   learningRate = updated_learningrate,
								   momentum = opt.momentum,
								  })
		---- if required then Net1 optimization----
		if opt.net1_freeze =='no' then
			gradParameters1:zero()
			feval_net1 = function(x)
				model.modules[1]:backward((batchInputs_source),dgradOutputs_mod2)
				return  gradOutputs_mod1, gradParameters1
				end
			optim.sgd(feval_net1, parameters1, {
								   learningRates = learningRates_Net1,
								   weightDecays = weightDecays_Net1,
								   learningRate = updated_learningrate,
								   momentum = opt.momentum,
								  })
		end


		local prob = softmax:forward({outputs5[1], outputs5[2]})
		local prob_sr = prob[1]
		local prob_tg = prob[2]
		local _, max_ids_sr = torch.max(prob_sr,2)
		local _, max_ids_tg = torch.max(prob_tg,2)

		local log_prob = logsoftmax:forward({outputs5[1]:clone(), outputs5[2]:clone()})
		local log_prob_sr = log_prob[1]
		local log_prob_tg = log_prob[2]
		local train_acc = check_accuracy(prob_sr, label)
        -- average classification loss ---------
		avg_loss = avg_loss + err/opt.batchSize
		avg_loss_cross = avg_loss_cross + criterion:forward(outputs5[1], label)
		-- average aleatoric source loss ---------
		avg_aleatoric_loss_sr = avg_aleatoric_loss_sr + err_aleatoric_cl_sr
        -- average aleatoric classification unceratinty for source -------
        avg_aleatoric_uncertainty_sr = avg_aleatoric_uncertainty_sr + torch.mean(outputsV[1])
        -- average aleatoric classification unceratinty for target -------
        avg_aleatoric_uncertainty_tg = avg_aleatoric_uncertainty_tg + torch.mean(outputsV[2])
         -- average aleatoric domain loss for source ---------
		avg_aleatoric_domain_loss_sr = avg_aleatoric_domain_loss_sr + err_aleatoric_dom_source
         -- average aleatoric domain loss for target ---------
		avg_aleatoric_domain_loss_tg = avg_aleatoric_domain_loss_tg + err_aleatoric_dom_target
        -- average aleatoric unceratinty for source-------
        avg_aleatoric_domain_uncertainty_sr = avg_aleatoric_domain_uncertainty_sr + torch.mean(outputsDV[1])
        -- average aleatoric unceratinty for target-------
        avg_aleatoric_domain_uncertainty_tg = avg_aleatoric_domain_uncertainty_tg + torch.mean(outputsDV[2])

		--- average training accuracy ----------
		avg_acc = avg_acc + train_acc
        --- average brier score ----------------
		avg_brier = avg_brier + torch.squeeze(((torch.pow(prob_sr - onehot:forward({label})[1], 2):mean(2))/31):mean(1))
        --- average entropy for source --------------------
		avg_entropy_sr = avg_entropy_sr - torch.squeeze(((torch.cmul(prob_sr, log_prob_sr)):sum(2)):mean(1))
		--- average entropy for target --------------------
		avg_entropy_tg = avg_entropy_tg - torch.squeeze(((torch.cmul(prob_tg, log_prob_tg)):sum(2)):mean(1))

		--- concentration for source domain
		local cnt_sr = torch.mul(prob_sr, opt.alpha_0_cls)
		--- concentration for target domain
		local cnt_tg = torch.mul(prob_tg, opt.alpha_0_cls)
 		--- average differential entropy for source --------------------
		avg_diff_entropy_sr = avg_diff_entropy_sr + torch.mean(torch.sum(lgamma(cnt_sr), 2) - lgamma(torch.sum(cnt_sr, 2)) - torch.cmul(31 - torch.sum(cnt_sr, 2), digamma(torch.sum(cnt_sr, 2))) - torch.sum(torch.cmul(torch.csub(cnt_sr, 1), digamma(cnt_sr)), 2))

		--- average differential entropy for target --------------------
		avg_diff_entropy_tg = avg_diff_entropy_tg + torch.mean(torch.sum(lgamma(cnt_tg), 2) - lgamma(torch.sum(cnt_tg, 2)) - torch.cmul(31 - torch.sum(cnt_tg, 2), digamma(torch.sum(cnt_tg, 2))) - torch.sum(torch.cmul(torch.csub(cnt_tg, 1), digamma(cnt_tg)), 2))

		--- average dirichlet variance for source --------------------
		local con0_sr = torch.expand(torch.sum(cnt_sr, 2):view(opt.batchSize, 1), opt.batchSize, 31)
		local var_sr = torch.cdiv(torch.cmul(cnt_sr, torch.csub(con0_sr, cnt_sr)), torch.cmul(torch.pow(con0_sr, 2), torch.add(con0_sr, 1)))
		avg_dir_var_sr = avg_dir_var_sr + torch.mean(var_sr:gather(2, max_ids_sr:view(max_ids_sr:size()[1], 1)))

		--- average dirichlet variance for target --------------------
		local con0_tg = torch.expand(torch.sum(cnt_tg, 2):view(opt.batchSize, 1), opt.batchSize, 31)
		local var_tg = torch.cdiv(torch.cmul(cnt_tg, torch.csub(con0_tg, cnt_tg)), torch.cmul(torch.pow(con0_tg, 2), torch.add(con0_tg, 1)))
		avg_dir_var_tg = avg_dir_var_tg + torch.mean(var_tg:gather(2, max_ids_tg:view(max_ids_tg:size()[1], 1)))

		train_acc = nil
		err = nil
		smooth_label = nil
		err_aleatoric_cl_sr = nil
		err_aleatoric_cl_tg = nil
        err_aleatoric_dom_source = nil
        err_aleatoric_dom_target = nil
		count = count + 1
	end
	epoch = epoch + 1
	return avg_loss/count, avg_loss_cross/count, avg_acc/count, avg_brier/count, avg_entropy_sr/count, avg_entropy_tg/count, avg_diff_entropy_sr/count, avg_diff_entropy_tg/count, avg_dir_var_sr/count, avg_dir_var_tg/count, avg_aleatoric_loss_sr/count,  avg_aleatoric_uncertainty_sr/count, avg_aleatoric_uncertainty_tg/count, avg_aleatoric_domain_loss_sr/count, avg_aleatoric_domain_loss_tg/count, avg_aleatoric_domain_uncertainty_sr/count, avg_aleatoric_domain_uncertainty_tg/count
end

--==============================Testing===================================================================
function test()
	-- disable flips, dropouts and batch normalization
	net1:evaluate()
	net2:evaluate()
	net3:evaluate()
	netB:evaluate()
	net4:evaluate()
	net5:evaluate()
    netV:evaluate()
	netD:evaluate()
	netDL:evaluate()
	netDV:evaluate()
	----------------scores---------------------------
	local test_err = 0
	local test_err_cross = 0
	local test_acc = 0
	local test_brier_score = 0
	local test_entropy = 0
	local test_var = 0
	local test_diff_entropy = 0
	local count = 0
    local test_aleatoric_loss = 0
    local test_aleatoric_uncertainty = 0
	local test_aleatoric_domain_loss_tg = 0
    local test_aleatoric_domain_uncertainty_tg = 0
	opt.Test_batchSize = opt.batchSize
	----------------testing----------------------------
	for i = 1,dataTest:size(), opt.Test_batchSize do
		data_tm:reset(); data_tm:resume()
		opt.start_Batch_IndexTest=i
		local batchInputs_test, validLabel = dataTest:getBatch(opt.start_Batch_IndexTest,dataTest:size())
		local probs_sims = torch.CudaTensor(validLabel:size()[1], 31):zero()
		local log_probs_sims = torch.CudaTensor(validLabel:size()[1], 31):zero()
		local outputs_sims = torch.Tensor(validLabel:size()[1], 31):zero()
		local outputsDL_sims = torch.Tensor(validLabel:size()[1], 2):zero()
		local smooth_factor = torch.Tensor(validLabel:size()[1], 31):fill((opt.alpha_0_cls - 31)/opt.alpha_0_cls)
		local bias = torch.Tensor(validLabel:size()[1], 31):fill(1/opt.alpha_0_cls)
		local variance_sims = torch.Tensor(validLabel:size()[1], 1):zero()
        local domain_variance_sims = torch.Tensor(validLabel:size()[1], 1):zero()
		if opt.gpu >=0 then
			probs_sims = probs_sims:cuda()
			outputs_sims = outputs_sims:cuda()
			batchInputs_test = batchInputs_test:cuda()
			validLabel = validLabel:cuda()
            domain_variance_sims = domain_variance_sims:cuda()
			outputsDL_sims = outputsDL_sims:cuda()
			variance_sims = variance_sims:cuda()
			log_probs_sims = log_probs_sims:cuda()
			smooth_factor = smooth_factor:cuda()
			bias = bias:cuda()
		end
		for j = 1, opt.num_sim do
			----------input forward-------------------------------
			local outputs1 = net1:forward({batchInputs_test:cuda()})
			local outputs2 = net2:forward(outputs1)
			local outputs3 = net3:forward(outputs2)
			local outputsB = netB:forward(outputs3)
			local outputs4 =  net4:forward(outputsB)
            local outputs5 = net5:forward(outputs4)  -- predicted class logits
            local outputsV = netV:forward(outputs4)  -- predicted variance
			local outputsD = netD:forward(outputsB)
			local outputsDL = netDL:forward(outputsD)  -- predicted domain logits
			local outputsDV = netDV:forward(outputsD)  -- predicted domain variance

			local probs = softmax:forward({outputs5[1]})[1]
			local log_probs = logsoftmax:forward({outputs5[1]})[1]
			probs_sims:add(probs)
			outputs_sims:add(outputs5[1])
            domain_variance_sims:add(outputsDV[1])
			outputsDL_sims:add(outputsDL[1])
			log_probs_sims:add(log_probs)
			variance_sims:add(outputsV[1])
		end

		------------loss and accuracy per model---------------
		local avg_probs = probs_sims/opt.num_sim
		local avg_log_probs = log_probs_sims/opt.num_sim
		local avg_logits = outputs_sims/opt.num_sim
		local _, avg_max_ids = torch.max(avg_probs, 2)
         ---------- classification Loss -------------------------------
		local smooth_label = torch.add(torch.cmul(onehot:forward({validLabel})[1], smooth_factor), bias)
		--- all the labels are possible as labels are smoothed
		local all_label = torch.CudaTensor(31)
		for c = 1, 31 do
			all_label[c] = c
		end

		err = 0
		for b = 1, validLabel:size()[1] do
			local criterion_weighted = nn.CrossEntropyCriterion(smooth_label[b])
			criterion_weighted:cuda()
			local temp_logits = avg_logits[b]:view(1, avg_logits[b]:size()[1])
			err = err + criterion_weighted:forward(torch.expand(temp_logits, temp_logits:size()[2], 31) , all_label)
			criterion_weighted = nil
		end
		test_err = test_err + err/(validLabel:size()[1])
		test_err_cross = test_err_cross + criterion:forward(avg_logits, validLabel)
        ------------- classification accuracy -------------------
		local test_batch_acc = check_accuracyTest(avg_probs, validLabel)
		test_acc = test_acc + test_batch_acc
		------------- brier score---------------------------------
		test_brier_score = test_brier_score + torch.squeeze(((torch.pow(avg_probs - onehot:forward({validLabel})[1], 2):mean(2))/31):mean(1))
		------------- entropy --------------------------------------
		test_entropy = test_entropy - torch.squeeze(((torch.cmul(avg_probs, avg_log_probs)):sum(2)):mean(1))

		--- concentration ------------
		local cnt = torch.mul(avg_probs, opt.alpha_0_cls)
 		--- average differential entropy  --------------------
		test_diff_entropy = test_diff_entropy + torch.mean(torch.sum(lgamma(cnt), 2) - lgamma(torch.sum(cnt, 2)) - torch.cmul(31 - torch.sum(cnt, 2), digamma(torch.sum(cnt, 2))) - torch.sum(torch.cmul(torch.csub(cnt, 1), digamma(cnt)), 2))

		--- average dirichlet variance for source --------------------
		local con0 = torch.expand(torch.sum(cnt, 2):view(validLabel:size()[1], 1), validLabel:size()[1], 31)
		local var = torch.cdiv(torch.cmul(cnt, torch.csub(con0, cnt)), torch.cmul(torch.pow(con0, 2), torch.add(con0, 1)))
		test_var = test_var + torch.mean(var:gather(2, avg_max_ids:view(var:size()[1], 1)))


        ------- aleatoric domain loss ---------------
        test_aleatoric_domain_loss_tg = test_aleatoric_domain_loss_tg + criterion_bayesian_dis:forward({outputsDL_sims/opt.num_sim, torch.Tensor(validLabel:size()[1]):fill(2), domain_variance_sims/opt.num_sim, torch.CudaTensor(validLabel:size()[1]):fill(1), torch.CudaTensor(1):fill(opt.num_sim)})[1]
        ------- aleatoric domain uncertainty --------
        test_aleatoric_domain_uncertainty_tg = test_aleatoric_domain_uncertainty_tg + torch.mean(domain_variance_sims)/opt.num_sim

		------- aleatoric classification loss ---------------
        test_aleatoric_loss = test_aleatoric_loss + criterion_bayesian_class:forward({outputs_sims/opt.num_sim, validLabel, variance_sims/opt.num_sim, torch.CudaTensor(validLabel:size()[1]):fill(1), torch.CudaTensor(1):fill(opt.num_sim)})[1]
        ------- aleatoric classification uncertainty --------
        test_aleatoric_uncertainty = test_aleatoric_uncertainty + torch.mean(variance_sims)/opt.num_sim

		test_batch_acc = nil
		err = nil
		smooth_label = nil
		count = count + 1
		confusion:batchAdd(avg_probs, validLabel)
	end
	confusion:updateValids()
	test_accuracy = confusion.totalValid
	if not acc_Logger then
		confusion:zero()
	end
	return test_accuracy, test_err/count, test_err_cross/count, test_acc/dataTest:size(), test_brier_score/count, test_entropy/count, test_diff_entropy/count, test_var/count, test_aleatoric_loss/count, test_aleatoric_uncertainty/count, test_aleatoric_domain_loss_tg/count, test_aleatoric_domain_uncertainty_tg/count
end

function save_html(train_error, train_loss_cross, train_acc, train_brier, train_entropy_sr, train_entropy_tg, train_diff_entropy_sr, train_diff_entropy_tg, train_var_sr, train_var_tg, train_aleatoric_loss_sr, train_aleatoric_uncertainty_sr, train_aleatoric_uncertainty_tg, train_aleatoric_domain_loss_sr, train_aleatoric_domain_loss_tg, train_aleatoric_domain_uncertainty_sr, train_aleatoric_domain_uncertainty_tg, test_accuracy, test_error, test_err_cross, test_acc, test_brier, test_entropy, test_diff_entropy, test_var, test_aleatoric_loss, test_aleatoric_uncertainty, test_aleatoric_domain_loss, test_aleatoric_domain_uncertainty)

	max_accuracy = math.max(max_accuracy, test_accuracy)

	if acc_Logger then
		paths.mkdir(opt.save)
		style = {'-', '-'}
		---------accuracy logger-----------
		acc_Logger:add{train_acc, test_acc}
		acc_Logger:style(style)
		acc_Logger:plot()
		---------brier logger-----------
		brier_Logger:add{train_brier, test_brier}
		brier_Logger:style(style)
		brier_Logger:plot()
		---------entropy logger-----------
		entropy_Logger:add{train_entropy_sr, train_entropy_tg, test_entropy}
		entropy_Logger:style({'-', '-', '-'})
		entropy_Logger:plot()
		---------differential entropy logger-----------
		entropy_diff_Logger:add{train_diff_entropy_sr, train_diff_entropy_tg, test_diff_entropy}
		entropy_diff_Logger:style({'-', '-', '-'})
		entropy_diff_Logger:plot()
		---------var logger-----------
		var_Logger:add{train_var_sr, train_var_tg, test_var}
		var_Logger:style({'-', '-', '-'})
		var_Logger:plot()
		---------error logger-----------
		error_Logger:add{train_error, test_error, train_loss_cross, test_err_cross}
		var_Logger:style({'-', '-', '-', '-'})
		error_Logger:plot()
        ---------domain error aleatoric logger-----------
		error_domain_aleatoric_Logger:add{train_aleatoric_domain_loss_sr, train_aleatoric_domain_loss_tg, test_aleatoric_domain_loss}
		error_domain_aleatoric_Logger:style({'-', '-', '-'})
		error_domain_aleatoric_Logger:plot()
        -----aleatoric domain uncertainty logger---------
		aleatoric_domain_uncertainty_Logger:add{train_aleatoric_domain_uncertainty_sr, train_aleatoric_domain_uncertainty_tg, test_aleatoric_domain_uncertainty}
		aleatoric_domain_uncertainty_Logger:style({'-', '-', '-'})
		aleatoric_domain_uncertainty_Logger:plot()
		---------classification error aleatoric logger-----------
		error_aleatoric_Logger:add{train_aleatoric_loss_sr, test_aleatoric_loss}
		error_aleatoric_Logger:style({'-', '-'})
		error_aleatoric_Logger:plot()
        -----aleatoric classification uncertainty logger---------
		aleatoric_uncertainty_Logger:add{train_aleatoric_uncertainty_sr, train_aleatoric_uncertainty_tg, test_aleatoric_uncertainty}
		aleatoric_uncertainty_Logger:style({'-', '-', '-'})
		aleatoric_uncertainty_Logger:plot()
		if paths.filep(opt.save..'/acc.log.eps') then
			------accu--------
			local base64im_acc
			do
				os.execute(('convert -density 200 %s/acc.log.eps %s/acc.png'):format(opt.save,opt.save))
				os.execute(('openssl base64 -in %s/acc.png -out %s/acc.base64'):format(opt.save,opt.save))
				local f = io.open(opt.save..'/acc.base64')
				if f then base64im_acc = f:read'*all' end
			end
			-------brier----------
			local base64im_brier
			do
				os.execute(('convert -density 200 %s/brier.log.eps %s/brier.png'):format(opt.save,opt.save))
				os.execute(('openssl base64 -in %s/brier.png -out %s/brier.base64'):format(opt.save,opt.save))
				local f = io.open(opt.save..'/brier.base64')
				if f then base64im_brier = f:read'*all' end
			end
			------entropy----------
			local base64im_entropy
			do
				os.execute(('convert -density 200 %s/entropy.log.eps %s/entropy.png'):format(opt.save,opt.save))
				os.execute(('openssl base64 -in %s/entropy.png -out %s/entropy.base64'):format(opt.save,opt.save))
				local f = io.open(opt.save..'/entropy.base64')
				if f then base64im_entropy = f:read'*all' end
			end
			------entropy----------
			local base64im_diff_entropy
			do
				os.execute(('convert -density 200 %s/diff_entropy.log.eps %s/diff_entropy.png'):format(opt.save,opt.save))
				os.execute(('openssl base64 -in %s/diff_entropy.png -out %s/diff_entropy.base64'):format(opt.save,opt.save))
				local f = io.open(opt.save..'/diff_entropy.base64')
				if f then base64im_diff_entropy = f:read'*all' end
			end
			------entropy----------
			local base64im_var
			do
				os.execute(('convert -density 200 %s/var.log.eps %s/var.png'):format(opt.save,opt.save))
				os.execute(('openssl base64 -in %s/var.png -out %s/var.base64'):format(opt.save,opt.save))
				local f = io.open(opt.save..'/var.base64')
				if f then base64im_var = f:read'*all' end
			end
			-------error-------
			local base64im_error
			do
				os.execute(('convert -density 200 %s/error.log.eps %s/error.png'):format(opt.save,opt.save))
				os.execute(('openssl base64 -in %s/error.png -out %s/error.base64'):format(opt.save,opt.save))
				local f = io.open(opt.save..'/error.base64')
				if f then base64im_error = f:read'*all' end
			end
            -------aleatoric domain error -------
			local base64im_error_domain_aleatoric
			do
				os.execute(('convert -density 200 %s/error_domain_aleatoric.log.eps %s/error_domain_aleatoric.png'):format(opt.save,opt.save))
				os.execute(('openssl base64 -in %s/error_domain_aleatoric.png -out %s/error_domain_aleatoric.base64'):format(opt.save,opt.save))
				local f = io.open(opt.save..'/error_domain_aleatoric.base64')
				if f then base64im_error_domain_aleatoric = f:read'*all' end
			end
            -------aleatoric domain uncertainty -------
			local base64im_aleatoric_domain_uncertainty
			do
				os.execute(('convert -density 200 %s/aleatoric_domain_uncertainty.log.eps %s/aleatoric_domain_uncertainty.png'):format(opt.save,opt.save))
				os.execute(('openssl base64 -in %s/aleatoric_domain_uncertainty.png -out %s/aleatoric_domain_uncertainty.base64'):format(opt.save,opt.save))
				local f = io.open(opt.save..'/aleatoric_domain_uncertainty.base64')
				if f then base64im_aleatoric_domain_uncertainty = f:read'*all' end
			end
			            -------aleatoric domain error -------
			local base64im_error_aleatoric
			do
				os.execute(('convert -density 200 %s/error_aleatoric.log.eps %s/error_aleatoric.png'):format(opt.save,opt.save))
				os.execute(('openssl base64 -in %s/error_aleatoric.png -out %s/error_aleatoric.base64'):format(opt.save,opt.save))
				local f = io.open(opt.save..'/error_aleatoric.base64')
				if f then base64im_error_aleatoric = f:read'*all' end
			end
            -------aleatoric classification uncertainty -------
			local base64im_aleatoric_uncertainty
			do
				os.execute(('convert -density 200 %s/aleatoric_uncertainty.log.eps %s/aleatoric_uncertainty.png'):format(opt.save,opt.save))
				os.execute(('openssl base64 -in %s/aleatoric_uncertainty.png -out %s/aleatoric_uncertainty.base64'):format(opt.save,opt.save))
				local f = io.open(opt.save..'/aleatoric_uncertainty.base64')
				if f then base64im_aleatoric_uncertainty = f:read'*all' end
			end

			local file = io.open(opt.save..'/report.html','w')
			file:write('<h5>Training data size:  '..data:size()..'\n')
			file:write('<h5>Validation data size:  '..dataTest:size()..'\n')
			file:write('<h5>batchSize:  '..batchSize..'\n')
			file:write('<h5>Network upto conv3 is freeze:   '..opt.net1_freeze..'\n')
			file:write('<h5>Base Learning Rate:  '..opt.baseLearningRate..'\n')
			file:write('<h5>momentum:  '..opt.momentum..'\n')
			file:write('<h5>Seed :  '..opt.manual_seed..'\n')
			file:write('<h5>lamda :  '..opt.lamda..'\n')
			file:write('<h5>number of test Class :  '..opt.number_of_testclass..'\n')
			file:write'</table><pre>\n'
			file:write(tostring(confusion)..'\n')
			file:write(tostring(net4)..'\n')
			file:write('<h5>Max Accuracy :  '..max_accuracy..'\n')
			file:write'</pre></body></html>'
			file:write(([[
			<!DOCTYPE html>
			<html>
			<body>
			<title>%s - %s</title>
			<img src="data:image/png;base64,%s">
			<table>
			]]):format(opt.save,epoch,base64im_acc))
			file:write(([[
			<!DOCTYPE html>
			<html>
			<body>
			<title>%s - %s</title>
			<img src="data:image/png;base64,%s">
			<table>
			]]):format(opt.save,epoch,base64im_brier))
			file:write(([[
			<!DOCTYPE html>
			<html>
			<body>
			<title>%s - %s</title>
			<img src="data:image/png;base64,%s">
			<table>
			]]):format(opt.save,epoch,base64im_entropy))
			file:write(([[
			<!DOCTYPE html>
			<html>
			<body>
			<title>%s - %s</title>
			<img src="data:image/png;base64,%s">
			<table>
			]]):format(opt.save,epoch,base64im_diff_entropy))
			file:write(([[
			<!DOCTYPE html>
			<html>
			<body>
			<title>%s - %s</title>
			<img src="data:image/png;base64,%s">
			<table>
			]]):format(opt.save,epoch,base64im_var))
			file:write(([[
			<!DOCTYPE html>
			<html>
			<body>
			<title>%s - %s</title>
			<img src="data:image/png;base64,%s">
			<table>
			]]):format(opt.save,epoch,base64im_error))
            file:write(([[
			<!DOCTYPE html>
			<html>
			<body>
			<title>%s - %s</title>
			<img src="data:image/png;base64,%s">
			<table>
			]]):format(opt.save,epoch,base64im_error_domain_aleatoric))
            file:write(([[
			<!DOCTYPE html>
			<html>
			<body>
			<title>%s - %s</title>
			<img src="data:image/png;base64,%s">
			<table>
			]]):format(opt.save,epoch,base64im_aleatoric_domain_uncertainty))
			file:write(([[
			<!DOCTYPE html>
			<html>
			<body>
			<title>%s - %s</title>
			<img src="data:image/png;base64,%s">
			<table>
			]]):format(opt.save,epoch,base64im_error_aleatoric))
            file:write(([[
			<!DOCTYPE html>
			<html>
			<body>
			<title>%s - %s</title>
			<img src="data:image/png;base64,%s">
			<table>
			]]):format(opt.save,epoch,base64im_aleatoric_uncertainty))
			file:close()
		end
		confusion:zero()
	end

		--print('epoch',epoch)
        if prev_accuracy< test_accuracy then
		print('Model is saving')
		collectgarbage()
		net1:clearState()
		net2:clearState()
		net3:clearState()
		netB:clearState()
		net4:clearState()
		netD:clearState()
		net5:clearState()
		netV:clearState()
		netDL:clearState()
		netDV:clearState()


		print('Model is Saved')
		prev_accuracy = test_accuracy
		end
end

for i = 1, opt.max_epoch do
	------------------training----------------------------------
	train_loss, train_loss_cross, train_acc, train_brier, train_entropy_sr, train_entropy_tg, train_diff_entropy_sr, train_diff_entropy_tg, train_var_sr, train_var_tg, train_aleatoric_loss_sr, train_aleatoric_uncertainty_sr, train_aleatoric_uncertainty_tg, train_aleatoric_domain_loss_sr, train_aleatoric_domain_loss_tg, train_aleatoric_domain_uncertainty_sr, train_aleatoric_domain_uncertainty_tg = train()

	print("-----------------------------------------------------")
	print('Train_acc', round(train_acc, 4), 'Train_loss', round(train_loss, 4), 'Train_loss_cross', round(train_loss_cross, 4), 'Train_sr_entropy', round(train_entropy_sr, 5), 'Train_tg_entropy', round(train_entropy_tg, 5), 'Train_sr_diff_entropy', round(train_diff_entropy_sr, 5), 'Train_tg_diff_entropy', round(train_diff_entropy_tg, 5), 'Train_barier', round(train_brier, 6), 'Train_sr_var', round(train_var_sr, 5), 'Train_tg_var', round(train_var_tg, 5), 'Train_sr_var', round(train_var_sr, 5), 'Train_tg_var', round(train_var_tg, 5),  'Train_aleatoric_loss_sr', round(train_aleatoric_loss_sr, 5),  'Train_aleatoric_uncertainty_sr', round(train_aleatoric_uncertainty_sr, 5),  'Train_aleatoric_uncertainty_tg', round(train_aleatoric_uncertainty_sr, 5), 'Train_aleatoric_domain_loss_sr', round(train_aleatoric_domain_loss_sr, 5), 'Train_aleatoric_domain_loss_tg', round(train_aleatoric_domain_loss_tg, 5), 'Train_aleatoric_domain_uncertainty_sr', round(train_aleatoric_domain_uncertainty_sr, 5), 'Train_aleatoric_domain_uncertainty_tg', round(train_aleatoric_domain_uncertainty_tg, 5))
	-----------------testing------------------------------------
	collectgarbage()
    net1:clearState()
    net2:clearState()
    net3:clearState()
    netB:clearState()
    net4:clearState()
    netD:clearState()
    net5:clearState()
    netV:clearState()
    netDL:clearState()
    netDV:clearState()
     -- defined in util.lua
    -- torch.save(paths.concat(opt.save, 'net1_' .. epoch .. '.t7'), net1)
    -- torch.save(paths.concat(opt.save, 'net2_' .. epoch .. '.t7'), net2)
    -- torch.save(paths.concat(opt.save, 'net3_' .. epoch .. '.t7'), net3)
    -- torch.save(paths.concat(opt.save, 'netB_' .. epoch .. '.t7'), netB)
    -- torch.save(paths.concat(opt.save, 'net4_' .. epoch .. '.t7'), net4)
    -- torch.save(paths.concat(opt.save, 'net5_' .. epoch .. '.t7'), net5)
    -- torch.save(paths.concat(opt.save, 'netV_' .. epoch .. '.t7'), netV)
    -- torch.save(paths.concat(opt.save, 'netD_' .. epoch .. '.t7'), netD)
    -- torch.save(paths.concat(opt.save, 'netDL_' .. epoch .. '.t7'), netDL)
    -- torch.save(paths.concat(opt.save, 'netDV_' .. epoch .. '.t7'), netDV)


    if (epoch<200 and epoch%10==0) or (epoch>200 and epoch%5==0) then
    	test_accuracy, test_err, test_err_cross, test_acc, test_brier_score, test_entropy, test_diff_entropy, test_var, test_aleatoric_loss, test_aleatoric_uncertainty, test_aleatoric_domain_loss, test_aleatoric_domain_uncertainty = test()
    	print("-----------------------------------------------------")
    	print("Test_accu:", round(test_acc, 4), "Test_loss:", round(test_err, 4), "Test_loss_cross:", round(test_err_cross, 4), "Test_entropy:", round(test_entropy, 5), "Test_diff_entropy:", round(test_diff_entropy, 5), "Test_brier:", round(test_brier_score, 6), "Test_var:", round(test_var, 5), "Test_aleatoric_loss:", round(test_aleatoric_loss, 5), "Test_aleatoric_uncertainty:", round(test_aleatoric_uncertainty, 5),  "Test_aleatoric_loss:", round(test_aleatoric_domain_loss, 5), "Test_aleatoric_uncertainty:", round(test_aleatoric_domain_uncertainty, 5), "Test Accuracy", test_accuracy)
    	print("-----------------------------------------------------")

    	-----------------saving-----------------------------------------
    	save_html(train_loss, train_loss_cross, train_acc, train_brier, train_entropy_sr, train_entropy_tg, train_diff_entropy_sr, train_diff_entropy_tg, train_var_sr, train_var_tg, train_aleatoric_loss_sr, train_aleatoric_uncertainty_sr, train_aleatoric_uncertainty_tg, train_aleatoric_domain_loss_sr, train_aleatoric_domain_loss_tg, train_aleatoric_domain_uncertainty_sr, train_aleatoric_domain_uncertainty_tg,  test_accuracy, test_err, test_err_cross, test_acc, test_brier_score, test_entropy, test_diff_entropy, test_var, test_aleatoric_loss, test_aleatoric_uncertainty, test_aleatoric_domain_loss, test_aleatoric_domain_uncertainty)
    	collectgarbage()
    end
end


