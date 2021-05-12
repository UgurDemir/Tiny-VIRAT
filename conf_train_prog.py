from deepnn.util import mkdirs

opt = {
    'v': 1,

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
        'produce': ['down', 'prog'],
        'workers': 4,
        'norm': '-11', # -11 | 01
    },

    'scale_shift': {
        1:{'erange': ( 0, 40 ), 'nfade':0, 'batch_size': 64, 'prog_size':[4,  28,  28]},
        2:{'erange': (40, 60 ), 'nfade':2, 'batch_size': 32, 'prog_size':[8,  56,  56]}, 
        3:{'erange': (60, 500), 'nfade':2, 'batch_size': 8,  'prog_size':[16, 112, 112]}
    },

    'gan1': {
        'D_iter': 1,
        'T_G': 0,
        'T_D': 0,

        'netG': {
            'gpu': 'cuda:0',

            'model': {
                #'arch': 'g6', 'attn':None, 'ngf':32, 'nbf':16, 'nblock':4, 'nup':2, 'acto':'tanh', 'norm':'spec', 'conv_mode':'3D',
                #'arch': 'g6', 'attn':'selfcorr3', 'ngf':32, 'nbf':16, 'nblock':4, 'nup':2, 'conv_mode':'3D',
                #'arch': 'g6', 'attn':'selfcorr4', 'ngf':32, 'nbf':16, 'nblock':4, 'nup':2, 'conv_mode':'3D',
                'arch': 'pg1', 'ngf':24,
            },

            'optim': {'method': 'Adam', 'lr': 0.0001, 'betas': (0.5, 0.9)},
            'lr_schedule': {'method': 'ExponentialLR', 'gamma': 1.0},
            #'lr_schedule': {'method': 'MultiStepLR', 'milestones':[30, 60, 100], 'gamma': 0.5},
            'loss': {
                'l1': {'wt': 100.0},
                'adv': {'wt': 1.0},
                #'adv_seg': {'wt': 0.2},
                #'feat':{'wt':0.1, 'model':'i3d', 'layers':[('incep3c',1.0)], 'loss':'l1', 'model_path':'/home/ugur/Documents/weights/i3d/converted_i3d_rgb_charades.pt'},
                #'gram':{'wt':120, 'model':'i3d', 'layers':[('incep3c',1.0)], 'loss':'l1', 'model_path':'/home/ugur/Documents/weights/i3d/converted_i3d_rgb_charades.pt'},
                #'style':{'wt':1.0, 'model':'i3d', 'loss':'l1', 'model_path':'/home/ugur/Documents/weights/i3d/converted_i3d_rgb_charades.pt',
                #        'wt_feat':0.05, 'feat_layers':[('incep3c',1.0)],
                #        'wt_gram':120, 'gram_layers':[('incep3c',1.0)]
                #},
                #'tv': {'wt': 0.1},
            },
            #'resume': '_results/model/checkpoint_19.pth.tar'
        }, 

        'aa':1,

        
        'netD': {
            'gpu': 'cuda:0',

            'model':{'arch':'pd1', 'ndf':16, 'actc':'lrelu', 'acto':'sigmoid', 'norm':'spec', 'conv_mode':'3D'},
            'optim': {'method': 'Adam', 'lr': 0.0004, 'betas': (0.5, 0.9)},
            'lr_schedule': {'method': 'ExponentialLR', 'gamma': 1.0},
            'loss': {
                'wt': 1.0, 'mode':'std', 'loss_type':'BCEWithLogitsLoss', 'real':1.0, 'fake':0.0
            }
        },
        

        'bb':1
    },

    'cuda': True,
    'manual_seed': None,

    # Output directories
    'outdir': '_results/out',
    'modeldir': '_results/model',
    'logdir': '_results/log',
    'slurmdir': '_results/slurm',

    # Training duration
    'start_epoch':0, 
    'epochs': 500,

    # Logging frequencies
    'print_freq': 20, # [iter]
    'image_freq': 200, # [iter]
    'val_image_freq': 200, # 1: means save every validation images for each epoch [iter]
    'val_print_freq': 100, # [iter]
    'log_freq': 1, # Save network weigth distrubutinons [epoch]
    'save_freq': 2 # Model saving frequency [epoch]
}


def parse(create_dirs=True):
    global opt

    ##############################################
    # Set random seed for all environment
    import random
    import torch
    import torch.backends.cudnn as cudnn
    if opt['manual_seed'] is None:
        opt['manual_seed'] = random.randint(1, 10000)
    print("Random Seed: ", opt['manual_seed'])
    random.seed(opt['manual_seed'])
    torch.manual_seed(opt['manual_seed'])
    if opt['cuda']:
        torch.cuda.manual_seed_all(opt['manual_seed'])
    cudnn.benchmark = True
    ##############################################

    if create_dirs:
        if 'outdir' in opt: mkdirs(opt['outdir'])
        if 'modeldir' in opt: mkdirs(opt['modeldir'])
        if 'logdir' in opt : mkdirs(opt['logdir'])
        if 'slurmdir' in opt: mkdirs(opt['slurmdir'])

    return opt

if __name__ == "__main__":
    parse(True)
