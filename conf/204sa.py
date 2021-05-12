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
        'batch_size': 17
    },

    'reset_ac_out': 101,
    'norm_to_01': True, # Renormalize images to [0,1] for resnet variations
    'netA': {
        'gpu': 'cuda:0',

        'model':{'arch':'resnet101'},
        'optim': {'method': 'SGD', 'lr': 0.1, 'momentum':0.9, 'weight_decay':1e-3, 'dampening':0.9, 'nesterov':False},
        'lr_schedule': {'method': 'ExponentialLR', 'gamma': 0.999},
        'loss': {
            'loss_type':'CrossEntropyLoss',
            #'loss_type':'BCEWithLogitsLoss',
        },
        'resume': '/home/ugur/Documents/weights/resnet3d/converted_resnet-101-kinetics.pth'
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
            'resume': '/home/ugur/Documents/repo/vidres/_results/155p/model/chk_prog_sr_scale2_best.pth.tar'
        }
    },

    'cuda': True,
    'manual_seed': None,

    # Training duration
    'start_epoch':0, 
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
