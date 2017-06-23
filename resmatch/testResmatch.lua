#!/usr/bin/env luajit
--[[调用testResmatch的文件，还需要的文件有opts.lua, runner.lua, resmatch.lua
    其中opts.lua中增加了参数-imgl（左图), -imgr(右图)
]]--

require 'image'
require 'nn'
require 'cutorch'
require 'os'

require 'cunn'
require 'cudnn'
require 'torch'
require 'paths'

require '../libadcensus'
require '../libcv'
require '../libcuresmatch'

local opts = require 'opts'
local opt = opts.parse(arg)
local resmatch = require('resmatch')(opt)
--opt中需要输入例如:-m fast, -imgl 000000_10, -imgr 000000_10
--其中左右图像分别放在../storage/color0 和../storage/color1中
--resmatch:testResmatch(opt)

--for i = 10,33 do
--	print(i)
--end

for i = 2, 194 do
	img_path = '%06d_10'
--	img_path = '1_%s_RIGHT'
--	opt.imgl = '000001_10.png'
--	opt.imgr = '000001_10.png'
	opt.imgl = img_path:format(i)
	opt.imgr = img_path:format(i)
 	print(opt.imgl)
	print(opt.imgr)	
	resmatch:testResmatch(opt)
end
