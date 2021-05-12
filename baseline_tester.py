import torch

from deepnn.measure.psnr import PSNR
from deepnn.measure.ssim import SSIM
from deepnn.nets.network_wrapper import NetworkWrapper
from reader.loader import get_test_loader
from deepnn.util.avg_meter import AverageMeter
from deepnn.measure.accuracy import Accuracy

BATCH_SIZE = 32

ucfx4 = {
    'dataset':{
        'name': 'ucf101',
        #'name': 'hmdb51', 
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
        'scale_factor': 4,
        'upscale_mode': 'bicubic',
        'downscale_mode': 'bicubic',
        'produce': ['up', 'orig'],
        'workers': 4,
        'norm': '-11', # -11 | 01
        'batch_size': BATCH_SIZE
    }
}

ucfx8 = {
    'dataset':{
        'name': 'ucf101',
        #'name': 'hmdb51', 
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
        'produce': ['up', 'orig'],
        'workers': 4,
        'norm': '-11', # -11 | 01
        'batch_size': BATCH_SIZE
    }
}


hmdbx4 = {
    'dataset':{
        #'name': 'ucf101',
        'name': 'hmdb51', 
        #'root': '/home/ugur/Documents/dataset/ucf101',
        'root': '/home/ugur/Documents/dataset/hmdb51',
        'split': {
            #'train_file': 'reader/ucf101/train.txt',
            #'test_file': 'reader/ucf101/test.txt',
            'train_file': 'reader/hmdb51/1/train.txt',
            'test_file': 'reader/hmdb51/1/test.txt',
        },
        'scale_size': 128, 
        'crop_size': 112,
        'nframe': 16,
        'scale_factor': 4,
        'upscale_mode': 'bicubic',
        'downscale_mode': 'bicubic',
        'produce': ['up', 'orig'],
        'workers': 4,
        'norm': '-11', # -11 | 01
        'batch_size': BATCH_SIZE
    }
}

hmdbx8 = {
    'dataset':{
        #'name': 'ucf101',
        'name': 'hmdb51', 
        #'root': '/home/ugur/Documents/dataset/ucf101',
        'root': '/home/ugur/Documents/dataset/hmdb51',
        'split': {
            #'train_file': 'reader/ucf101/train.txt',
            #'test_file': 'reader/ucf101/test.txt',
            'train_file': 'reader/hmdb51/1/train.txt',
            'test_file': 'reader/hmdb51/1/test.txt',
        },
        'scale_size': 128, 
        'crop_size': 112,
        'nframe': 16,
        'scale_factor': 8,
        'upscale_mode': 'bicubic',
        'downscale_mode': 'bicubic',
        'produce': ['up', 'orig'],
        'workers': 4,
        'norm': '-11', # -11 | 01
        'batch_size': BATCH_SIZE
    }
}

device = torch.device("cuda")
psnr = PSNR(peak=1.0)
ssim = SSIM()

def test(name, opt):
    loader = get_test_loader(**opt['dataset'])

    psnr_meter, ssim_meter = AverageMeter(), AverageMeter()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            up, gt = batch[0].to(device), batch[1].to(device)

            # Denorm the output so that they lie between 0 and 1
            up_dn = loader.dataset.denormalize(up)
            gt_dn = loader.dataset.denormalize(gt)
            # Test
            b, c, t, h, w = gt.size()
            up_batch = up_dn.permute(0,2,1,3,4).contiguous().view(b*t, c, h, w)
            gt_batch = gt_dn.permute(0,2,1,3,4).contiguous().view(b*t, c, h, w)
            pval = psnr(up_batch.data, gt_batch.data)
            ssim_val = ssim(up_batch.data, gt_batch.data)
            psnr_meter.update(pval, up_batch.size(0))
            ssim_meter.update(ssim_val, up_batch.size(0))
            
            #if i % 5 == 0:
            #    print('test [{}/{}] acc: {acc:.3f} '.format(i, len(loader), acc=acc_meter.avg), flush=True)
    
    print(name + ' test psnr: {:.2f}, ssim: {:.2f}'.format(psnr_meter.avg, ssim_meter.avg), flush=True)


if __name__ == "__main__":
    test('ucfx4', ucfx4)
    test('ucfx8', ucfx8)
    test('hmdbx4', hmdbx4)
    test('hmdbx8', hmdbx8)