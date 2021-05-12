import torch

from deepnn.measure.psnr import PSNR
from deepnn.measure.ssim import SSIM
from deepnn.nets.network_wrapper import NetworkWrapper
from reader.loader import get_test_loader
from deepnn.util.avg_meter import AverageMeter
from deepnn.measure.accuracy import Accuracy

opt0 = {
    'dataset':{
        #'name': 'hmdb51', 
        'name': 'ucf101',
        'root': '/home/ugur/Documents/dataset/ucf101',
        #'root': '/home/ugur/Documents/dataset/hmdb51',
        'split': {
            'train_file': 'reader/ucf101/train.txt',
            'test_file': 'reader/ucf101/test.txt',
            #'train_file': 'reader/hmdb51/1/train.txt',
            #'test_file': 'reader/hmdb51/1/test.txt',
        },
        'scale_size': 128, 
        'crop_size': 112,
        'nframe': 16,
        'scale_factor': 8,
        'upscale_mode': 'bicubic',
        'downscale_mode': 'bicubic',
        'produce': ['down', 'orig'],
        'workers': 4,
        'norm': '-11', # -11 | 01
        'batch_size': 18
    },

    'netG': {
        'gpu': 'cuda:0',
        'model': {'arch': 'g6', 'attn':None, 'ngf':32, 'nbf':16, 'nblock':4, 'nup':3, 'acto':'tanh', 'norm':'spec', 'conv_mode':'3D'},
        'resume': '/home/ugur/Documents/exps/vidres/86/_results/model/chk_sr_best.pth.tar'
    },

    'cuda': True,
}

opt155p = {
    'dataset':{
        #'name': 'hmdb51', 
        'name': 'ucf101',
        'root': '/home/ugur/Documents/dataset/ucf101',

        'scale_size': 128, 
        'crop_size': 112,
        'nframe': 16,
        'scale_factor': 4,
        'upscale_mode': 'bicubic',
        'downscale_mode': 'bicubic',
        'produce': ['down', 'orig'],
        'workers': 8,
        'norm': '-11', # -11 | 01
	'batch_size': 64
    },
    
    'netG': {
        'gpu': 'cuda:0',
        'model': {'arch': 'pg2', 'mode': '3D', 'scale_factor':4,  'ngf':32, 'nbf':32,  'norm':None, 'conv_mode':'3D'},
        'resume': '/home/ugur/Documents/repo/vidres/_results/155p/model/chk_prog_sr_scale2_best.pth.tar'
    }
}

opt158p = {
    'dataset':{
        #'name': 'hmdb51', 
        'name': 'ucf101',
        'root': '/home/ugur/Documents/dataset/ucf101',

        'scale_size': 128, 
        'crop_size': 112,
        'nframe': 16,
        'scale_factor': 8,
        'upscale_mode': 'bicubic',
        'downscale_mode': 'bicubic',
        'produce': ['down', 'orig'],
        'workers': 8,
        'norm': '-11', # -11 | 01
	'batch_size': 64
    },
    
    'netG': {
        'gpu': 'cuda:0',
        'model': {'arch': 'pg2', 'mode': '3D', 'scale_factor':8,  'ngf':32, 'nbf':32,  'norm':None, 'conv_mode':'3D'},
        'resume': '/home/ugur/Documents/repo/vidres/_results/158p/model/chk_prog_sr_scale3_best.pth.tar'
    }
}
###############
opt159p = {
    'dataset':{
        'name': 'hmdb51', 
        #'name': 'ucf101',
        'root': '/home/ugur/Documents/dataset/hmdb51',

        'scale_size': 128, 
        'crop_size': 112,
        'nframe': 16,
        'scale_factor': 4,
        'upscale_mode': 'bicubic',
        'downscale_mode': 'bicubic',
        'produce': ['down', 'orig'],
        'workers': 8,
        'norm': '-11', # -11 | 01
	'batch_size': 64
    },
    
    'netG': {
        'gpu': 'cuda:0',
        'model': {'arch': 'pg2', 'mode': '3D', 'scale_factor':4,  'ngf':32, 'nbf':32,  'norm':None, 'conv_mode':'3D',},
        'resume': '/home/ugur/Documents/repo/vidres/_results/159p/model/chk_prog_sr_scale2_best.pth.tar'
    }
}

opt162p = {
    'dataset':{
        'name': 'hmdb51', 
        #'name': 'ucf101',
        'root': '/home/ugur/Documents/dataset/hmdb51',

        'scale_size': 128, 
        'crop_size': 112,
        'nframe': 16,
        'scale_factor': 8,
        'upscale_mode': 'bicubic',
        'downscale_mode': 'bicubic',
        'produce': ['down', 'orig'],
        'workers': 8,
        'norm': '-11', # -11 | 01
	'batch_size': 64
    },
    
    'netG': {
        'gpu': 'cuda:0',
        'model': {'arch': 'pg2', 'mode': '3D', 'scale_factor':8,  'ngf':32, 'nbf':32,  'norm':None, 'conv_mode':'3D'},
        'resume': '/home/ugur/Documents/repo/vidres/_results/162p/model/chk_prog_sr_scale3_best.pth.tar'
    }
}



device = torch.device("cuda")
psnr = PSNR(peak=1.0)
ssim = SSIM()

def test(name, opt):
    gen = NetworkWrapper(name='G', **opt['netG']).to(device)

    loader = get_test_loader(**opt['dataset'])

    gen.eval()
    psnr_meter, ssim_meter = AverageMeter(), AverageMeter()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            ctx, gt = batch[0].to(device), batch[1].to(device)

            # Forward G
            rec = gen(ctx)

            # Denorm the output so that they lie between 0 and 1
            rec_dn = loader.dataset.denormalize(rec)
            gt_dn = loader.dataset.denormalize(gt)

            # Test
            b, c, t, h, w = gt.size()
            rec_batch = rec_dn.permute(0,2,1,3,4).contiguous().view(b*t, c, h, w)
            gt_batch = gt_dn.permute(0,2,1,3,4).contiguous().view(b*t, c, h, w)
            pval = psnr(rec_batch.data, gt_batch.data)
            ssim_val = ssim(rec_batch.data, gt_batch.data)
            psnr_meter.update(pval, rec_batch.size(0))
            ssim_meter.update(ssim_val, rec_batch.size(0))
            
            #if i % 5 == 0:
            #    print('test [{}/{}] acc: {acc:.3f} '.format(i, len(loader), acc=acc_meter.avg), flush=True)
            
    
    print(name + ' test psnr: {:.2f}, ssim: {:.2f}'.format(psnr_meter.avg, ssim_meter.avg), flush=True)


if __name__ == "__main__":
    #test('155p', opt155p)
    #test('158p', opt158p)
    #test('159p', opt159p)
    test('162p', opt162p)

