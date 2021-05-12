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
        'produce': ['down','label'],
        'workers': 4,
        'norm': '-11', # -11 | 01
        'batch_size': 24
    },


    'netG': {
        'gpu': 'cuda:0',
        'model': {'arch': 'pg2', 'mode': '3D', 'scale_factor':4,  'ngf':32, 'nbf':32,  'norm':None, 'conv_mode':'3D'},
        'load': '/home/ugur/Documents/repo/vidres/_results/222l/model/gan/chk_sr_7.pth.tar'
    },


    'bb':1,

    'reset_ac_out': 101,
    'norm_to_01': False, # Renormalize images to [0,1] for resnet variations
    'netA': {
        'gpu': 'cuda:0',

        'model':{'arch':'i3d'},
        'optim': {'method': 'Adam', 'lr': 0.0002},
        'lr_schedule': {'method': 'ExponentialLR', 'gamma': 1.0},
        'loss': {
            'loss_type':'CrossEntropyLoss',
            #'loss_type':'BCEWithLogitsLoss',
        },
        'resume': '/home/ugur/Documents/weights/i3d/converted_i3d_rgb_charades.pt'
    },

    'cuda': True,
    'manual_seed': None,

    # Training duration
    'start_epoch':0, 
    'epochs': 500,

    # Logging frequencies
    'print_freq': 20, # [iter]
    'val_print_freq': 500, # [iter]
    'log_freq': 1, # Save network weigth distrubutinons [epoch]
    'save_freq': 20 # Model saving frequency [epoch]
}

def get():
    global opt
    return opt