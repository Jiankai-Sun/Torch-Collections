#!/usr/bin/env luajit
--[[
This file is used to test /data/jack/AutoNav_Data/ConferenceHall-3/ Dataset
]]--
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

require 'lfs'

local opts = require 'opts'
local opt = opts.parse(arg)
local resmatch = require('resmatch')(opt)
--opt中需要输入例如:-m fast, -imgl 000000_10, -imgr 000000_10
--其中左右图像分别放在../storage/color0 和../storage/color1中
--resmatch:testResmatch(opt)


--[[
–获取路径
function stripfilename(filename)
return string.match(filename, “(.+)/[^/]*%.%w+$”) — *nix system
–return string.match(filename, “(.+)\\[^\\]*%.%w+$”) — windows
end

–获取文件名
function strippath(filename)
return string.match(filename, “.+/([^/]*%.%w+)$”) — *nix system
–return string.match(filename, “.+\\([^\\]*%.%w+)$”) — *nix system
end

–去除扩展名
function stripextension(filename)
local idx = filename:match(“.+()%.%w+$”)
if(idx) then
return filename:sub(1, idx-1)
else
return filename
end
end

–获取扩展名
function getextension(filename)
return filename:match(“.+%.(%w+)$”)
end
]]--

function mysplit(inputstr, sep)
        if sep == nil then
                sep = "%s"
        end
        local t={} ; i=1
        for str in string.gmatch(inputstr, "([^"..sep.."]+)") do
                t[i] = str
                i = i + 1
        end
        return t
end

path = '/data/jack/AutoNav_Data/ConferenceHall-3'
subpath1 = path..'/ConferenceHall-3-1'
start_time = os.clock()

for file in lfs.dir(subpath1) do
	if file:match(".+%.(%w+)$") == 'png' then
		idx=file:match(".+()%.%w+$")
		substr=file:sub(1, idx-1)
		split = mysplit(substr, "_")
		opt.imgl = substr
		opt.imgr = split[1].."_"..split[2].."_"..split[3].."_".."RIGHT"
		print(opt.imgl)
		print(opt.imgr)	
		local x = os.clock()
		resmatch:testResmatch(opt)
		print(string.format("elapsed time per picture: %.2f\n", os.clock()-x))
	end
end

print(string.format("Elapsed time for all pictures: %.2f\n", os.clock()-start_time))

