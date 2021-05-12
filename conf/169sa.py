opt = {
    'v': 1,

    'dataset':{
        #'name': 'hmdb51', 
        'name': 'ucf101',
        #'name': 'diva',

        'scale_size': 128, 
        'crop_size': 112,
        'nframe': 16,
        'scale_factor': 4,
        'upscale_mode': 'bicubic',
        'downscale_mode': 'bicubic',
        'produce': ['down', 'orig', 'label'],
        'workers': 8,
        'norm': '-11', # -11 | 01
        'batch_size': 14
    },

    #'reset_ac_out': 101,
    'netA': {
        'gpu': 'cuda:0',

        'model':{'arch':'i3d', 'num_classes':101},
        'optim': {'method': 'Adam', 'lr': 0.0002},
        'lr_schedule': {'method': 'ExponentialLR', 'gamma': 0.999},
        'loss': {
            'loss_type':'CrossEntropyLoss',
            #'loss_type':'BCEWithLogitsLoss',
        },
        #'resume': '/home/ugur/Documents/weights/i3d/converted_i3d_rgb_charades.pt'
        'resume': '/home/ugur/Documents/repo/vidres/_results/169sa/model/ac/chk_ac_39.pth.tar'
    },

    'gan1': {
        'D_iter': 1,
        'T_G': 0,
        'T_D': 0,

        'netG': {
            'gpu': 'cuda:0',

            'model': {
                #'arch': 'g6', 'attn':None, 'ngf':32, 'nbf':16, 'nblock':4, 'nup':2, 'acto':'tanh', 'norm':'spec', 'conv_mode':'3D',
                'arch': 'pg2', 'mode': '3D', 'scale_factor':4,  'ngf':32, 'nbf':32,  'norm':None, 'conv_mode':'3D',
            },

            'optim': {'method': 'Adam', 'lr': 0.0002, 'betas': (0.5, 0.9)},
            'lr_schedule': {'method': 'ExponentialLR', 'gamma': 1.0},
            'loss': {
                'l1': {'wt': 100.0},
                #'adv': {'wt': 1.0},
                'ac' : {'wt': 1.0, 'loss':'CrossEntropyLoss'},
                #'feat':{'wt':0.1, 'model':'i3d', 'layers':[('incep3c',1.0)], 'loss':'l1', 'model_path':'/home/ugur/Documents/weights/i3d/converted_i3d_rgb_charades.pt'},
                #'gram':{'wt':120, 'model':'vgg16', 'layers':[('pool_1',1.0),('pool_2',1.0)], 'loss':'l1'},
                #'style':{'wt':1.0, 'model':'vgg16', 'loss':'l2',
                #        'wt_feat':0.05, 'feat_layers':[('pool_1',1.0),('pool_2',1.0)],
                #        'wt_gram':120, 'gram_layers':[('pool_1',1.0),('pool_2',1.0)]
                #},
                #'tv': {'wt': 0.1},
            },
            #'resume': '/home/ugur/Documents/repo/vidres/_results/155p/model/chk_prog_sr_scale2_best.pth.tar'
            'resume': '/home/ugur/Documents/repo/vidres/_results/169sa/model/gan/chk_sr_39.pth.tar'
        }, 

        'aa':1,

        """
        'netD': {
            'gpu': 'cuda:0',

            'model':{'arch':'d3', 'ndf':16, 'nfc':256, 'nout':16, 'actc':'lrelu', 'acto':None, 'norm':'spec', 'conv_mode': '3D'},
            'optim': {'method': 'Adam', 'lr': 0.0004, 'betas': (0.5, 0.9)},
            'lr_schedule': {'method': 'ExponentialLR', 'gamma': 1.0},
            'loss': {
                'wt': 1.0, 'mode':'std', 'loss_type':'BCEWithLogitsLoss', 'real':1.0, 'fake':0.0
            },
            'resume': '/home/ugur/Documents/exps/vidres/75/chk_sr_best.pth.tar'
        },
        """
        
        'bb':1
    },

    'cuda': True,
    'manual_seed': None,

    # Training duration
    'start_epoch':40, 
    'epochs': 500,

    # Logging frequencies
    'print_freq': 20, # [iter]
    'image_freq': 5000, # [iter]
    'val_freq': 5, # [epoch]
    'val_image_freq': 200, # 1: means save every validation images for each epoch [iter]
    'val_print_freq': 40, # [iter]
    'log_freq': 1, # Save network weigth distrubutinons [epoch]
    'save_freq': 20 # Model saving frequency [epoch]
}


def get():
    global opt
    return opt
