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
        'batch_size': 32
    },

    'bb': 1,

    
    'netG': {
        'gpu': 'cuda:0',
        'model': {'arch': 'g6', 'attn':None, 'ngf':32, 'nbf':16, 'nblock':4, 'nup':2, 'acto':'tanh', 'norm':'spec', 'conv_mode':'3D'},
        'resume': '_results/75/chk_sr_best.pth.tar'
    },


    'bb':1,

    'reset_ac_out': 101,
    'netA': {
        'gpu': 'cuda:0',

        'model':{'arch':'i3d'},
        'optim': {'method': 'Adam', 'lr': 0.0002, 'betas':(0.5, 0.999)},
        'lr_schedule': {'method': 'ExponentialLR', 'gamma':0.999},
        'loss': {
            'loss_type':'CrossEntropyLoss'
        },
        'resume': '/home/ugur/Documents/weights/i3d/converted_i3d_rgb_charades.pt'
    },


    'cuda': True,
    'manual_seed': None,

    # Training duration
    'start_epoch':0, 
    'epochs': 500,

    # Logging frequencies
    'print_freq': 40, # [iter]
    'val_print_freq': 500, # [iter]
    'log_freq': 1, # Save network weigth distrubutinons [epoch]
    'save_freq': 20 # Model saving frequency [epoch]
}

def get():
    global opt
    return opt