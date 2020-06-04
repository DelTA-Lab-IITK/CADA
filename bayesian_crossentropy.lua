
 --Written by   Shanu Kumar
 --Copyright (c) 2019, Shanu Kumar [See LICENSE file for details]




require 'nn'
require 'nngraph'
require 'cudnn'
require 'cunn'
require 'Sampler'

local bayesian_crossentropy = {}
function bayesian_crossentropy.criterion(num_sims, dim)
    num_sims = num_sims or 100           -- Number of montecarlo simulations
    local pred_logit = nn.Identity()();  -- Prediction logits
    local true_label = nn.Identity()();  -- True label
    local var = nn.Identity()();         -- Predicted Variance
    local const = nn.Identity()();       -- Constant tensor containing 1 of batch size
    local sims = nn.Identity()();        -- Constant tensor containing no. of simulations
    local std = nn.Sqrt()(var)           -- Standard deviation
    -- variance depressor = exp(variance) - 1 ---------
    local variance_depressor = nn.Mean()(nn.CSubTable()({nn.Exp()(var), const}))
    ------ undistorted loss ---------------------------
    local undistorted_loss = nn.CrossEntropyCriterion()({pred_logit, true_label})
    ----- montecarlo simulations ----------------------
    local monte_carlo_results = {}
    for i = 1, num_sims do
        ----- distorted loss = Cross_entropy(pred_logit + sampled noise, true label)
        local distorted_loss = nn.CrossEntropyCriterion()({nn.CAddTable()({pred_logit, nn.Sampler(dim)(std)}), true_label})
        --- diff = Elu(undistorted loss - distorted loss) -----------
        table.insert(monte_carlo_results, nn.ELU(1)(nn.CSubTable()({undistorted_loss, distorted_loss})))
    end
    ---- variance loss = undistorted loss * mean(monte_carlo_results) ------
    local variance_loss = nn.CMulTable()({undistorted_loss, nn.CDivTable()({nn.CAddTable()(monte_carlo_results), sims})})
    ---- output = variance loss + undistorted loss + variance depressor ----
    local output = nn.CAddTable()({variance_loss, undistorted_loss, variance_depressor})

    return nn.gModule({pred_logit, true_label, var, const, sims}, {output})
end

return bayesian_crossentropy
