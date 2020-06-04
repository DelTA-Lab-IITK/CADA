-- Based on JoinTable module


 --Written by   Shanu Kumar
 --Copyright (c) 2019, Shanu Kumar [See LICENSE file for details]

require 'nn'

local Sampler, parent = torch.class('nn.Sampler', 'nn.Module')

function Sampler:__init(dim)
    parent.__init(self)
	self.dim = dim
end

function Sampler:updateOutput(input)
    self.eps = torch.randn(input:size(1), self.dim):type(input:type())
    self.output = self.output or self.output.new()
    self.output:resizeAs(self.eps):copy(self.eps)
    self.output:cmul(torch.expand(input, input:size(1), self.dim))
    return self.output
end

function Sampler:updateGradInput(input, gradOutput)
    self.gradInput = self.gradInput or input.new()
    self.gradInput:resizeAs(input)
    local gi = torch.cmul(self.eps, torch.expand(input, input:size(1), self.dim))
    gi:mul(0.5):cmul(gradOutput)
    self.gradInput:copy(torch.sum(gi, 2))
    return self.gradInput
end
