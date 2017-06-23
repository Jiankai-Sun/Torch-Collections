--接下来是ResMatch类

local M = {}

local ResMatch = torch.class('ResMatch', M)

function ResMatch:__init(opt)
   self.matcher = require('pipeline/matching')
   self.disp = require('pipeline/disparity')
   self.post = require('pipeline/post')
   self.refiner = require('pipeline/refinement')
   self.mcnet = require('networks/mc-models/'..opt.mc..'/'..opt.m):new(opt)
   --opt.mc = resmatch opt.m = fast/hybrid
   self.opt = opt
   print('===> Loading matching cost network...')
   self.checkpoint, self.optimState = self.mcnet:load(opt)
   print('===> Loaded! Network ')
   self.runner = require('runner')(self.mcnet, nil, opt)
end

function ResMatch:testResmatch(opt)
    torch.manualSeed(opt.seed)
    cutorch.manualSeed(opt.seed)
    cutorch.setDevice(tonumber(opt.gpu))
    print('dataset')
    n_tr = 0--194
    n_te = 1--195
    data = opt.storage
    image_0 = 'colored_0'
    image_1 = 'colored_1'
    nchannel = 3
--    height = 600
--    width = 800
    --height = 720
    --width = 1280
    disp_max = 228
    --height = 350
    height= 375
    --width = 1241
    width=1242

    x0 = torch.FloatTensor(1, nchannel, height, width):zero()
    x1 = torch.FloatTensor(1, nchannel, height, width):zero()
    metadata = torch.IntTensor(n_tr + n_te, 3):zero()

    img_path = '%s/%s/%s.png'
    img_0 = image.loadPNG(img_path:format(data, image_0, opt.imgl), nchannel, 'byte'):float()
    img_1 = image.loadPNG(img_path:format(data, image_1, opt.imgr), nchannel, 'byte'):float()

    img_height = img_0:size(2)
    img_width = img_0:size(3)
    img_0 = img_0:narrow(2, img_height - height + 1, height)
    img_1 = img_1:narrow(2, img_height - height + 1, height)

    img_0:add(-img_0:mean()):div(img_0:std())
    img_1:add(-img_1:mean()):div(img_1:std())

    --print(img_0)
    --print(x0.size())
    x0[{i,{},{},{1,img_width}}]:copy(img_0)
    x1[{i,{},{},{1,img_width}}]:copy(img_1)

    metadata[{i, 1}] = img_height
    metadata[{i, 2}] = img_width
    metadata[{i, 3}] = 1 - 1

    directions =  {1, -1}

    n_colors = nchannel

    img={}
    img.x_batch = torch.CudaTensor(2, n_colors, img_height, img_width)
    img.x_batch:resize(2, n_colors, x0:size(3), x0:size(4))
    img.x_batch[1]:copy(x0)
    img.x_batch[2]:copy(x1)

    cutorch.synchronize()
    sys.tic()

    pred = self.runner:predict(img, disp_max, directions, false)

    print('\n' .. 'Finished!' .. '\n')
    save_png(disp_max, pred, opt)
    --filename = ('../tmp/%s_%s_%s_pred.csv'):format(opt.ds, opt.m, opt.imgl)
    --save_2dTensor(pred[1][1], filename)
end

return M.ResMatch
