opt = { 
    'v': 1,

    'dataset':{
        'name': 'hmdb51', 
        #'name': 'ucf101',

        'scale_size': 128, 
        'crop_size': 112,
        'nframe': 16,
        'scale_factor': 8,
        'upscale_mode': 'bicubic',
        'downscale_mode': 'bicubic',
        'produce': ['up', 'prog'],
        'workers': 4,
        'norm': '-11', # -11 | 01
    },

    'scale_shift': {
        1:{'erange': (0 , 20), 'nfade':0, 'batch_size': 22, 'prog_size':[4,  28,  28]},
        2:{'erange': (20, 40), 'nfade':2, 'batch_size': 18, 'prog_size':[8,  56,  56]}, 
        3:{'erange': (40, 500), 'nfade':2, 'batch_size': 14,  'prog_size':[16, 112, 112]}
    },

    'gan1': {
        'D_iter': 1,
        'T_G': 0,
        'T_D': 0,

        'netG': {
            'gpu': 'cuda:0',

            'model': {
                'arch': 'pg5', 'nch':14, 'actenc':'lrelu', 'actdec':'lrelu', 'acto':'tanh', 'norm':'bn', 'conv_mode':'3D',
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
        }, 


    },

    'cuda': True,
    'manual_seed': None,

    # Training duration
    'start_epoch':0, 
    'epochs': 500,

    # Logging frequencies
    'print_freq': 20, # [iter]
    'image_freq': 500, # [iter]
    'val_image_freq': 200, # 1: means save every validation images for each epoch [iter]
    'val_print_freq': 100, # [iter]
    'log_freq': 1, # Save network weigth distrubutinons [epoch]
    'save_freq': 2 # Model saving frequency [epoch]
}


def get():
    global opt
    return opt
