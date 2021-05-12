opt = {
    'v': 1,

    'dataset':{
        'name': 'hmdb51', 


        'scale_size': 128, 
        'crop_size': 112,
        'nframe': 16,
        'scale_factor': 8,
        'upscale_mode': 'bicubic',
        'downscale_mode': 'bicubic',
        'produce': ['up', 'orig'],
        'workers': 8,
        'norm': '-11', # -11 | 01
        'batch_size': 10
    },

    'gan1': {
        'D_iter': 1,
        'T_G': 0,
        'T_D': 0,

        'netG': {
            'gpu': 'cuda:0',

            'model': {
                'arch': 'g8', 'nbf':16, 'act':'lrelu', 'acto':'tanh', 'norm':'spec', 'conv_mode':'3D',
                #'arch': 'g6', 'attn':'selfcorr3', 'ngf':32, 'nbf':16, 'nblock':4, 'nup':2, 'conv_mode':'3D',
                #'arch': 'g6', 'attn':'selfcorr4', 'ngf':32, 'nbf':16, 'nblock':4, 'nup':2, 'conv_mode':'3D',
            },

            'optim': {'method': 'Adam', 'lr': 0.0002, 'betas': (0.5, 0.9)},
            'lr_schedule': {'method': 'ExponentialLR', 'gamma': 1.0},
            #'lr_schedule': {'method': 'MultiStepLR', 'milestones':[30, 60, 100], 'gamma': 0.5},
            'loss': {
                'l1': {'wt': 1.0},
                #'adv': {'wt': 1.0},
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

        '''
        'netD': {
            'gpu': 'cuda:0',

            'model':{'arch':'d3', 'ndf':16, 'nfc':256, 'nout':16, 'actc':'lrelu', 'acto':None, 'norm':'spec', 'conv_mode': '3D'},
            'optim': {'method': 'Adam', 'lr': 0.0004, 'betas': (0.5, 0.9)},
            'lr_schedule': {'method': 'ExponentialLR', 'gamma': 1.0},
            'loss': {
                'wt': 1.0, 'mode':'std', 'loss_type':'BCEWithLogitsLoss', 'real':1.0, 'fake':0.0
            }
        },
        '''

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
    'print_freq': 40, # [iter]
    'image_freq': 5000, # [iter]
    'val_image_freq': 200, # 1: means save every validation images for each epoch [iter]
    'val_print_freq': 100, # [iter]
    'log_freq': 1, # Save network weigth distrubutinons [epoch]
    'save_freq': 20 # Model saving frequency [epoch]
}

def get():
    global opt
    return opt
