import importlib
from os.path import join

from deepnn.util import mkdirs

MACHINE_CONGIFS = {
    'desktop': {
        'ucf101': { 
            'root': '/home/ugur/Documents/dataset/ucf101',
        },

        'hmdb51': {
            'root': '/home/ugur/Documents/dataset/hmdb51',
        },
    },

    'crcv': {
        'ucf101': { 
            'root': '/home/c3-0/ugur/dataset/ucf101',
        },

        'hmdb51': {
            'root': '/home/c3-0/ugur/dataset/hmdb51',
        },
    },

    # Output directories
    'container_dir': '_results',
    'outdir': 'out',
    'modeldir': 'model',
    'ac_modeldir': 'model/ac',
    'gan_modeldir': 'model/gan',
    'logdir': 'log',
    'slurmdir': 'slurm',
}

def create_dirs(exp_name):
    outdir = join(MACHINE_CONGIFS['container_dir'], exp_name, MACHINE_CONGIFS['outdir'])
    modeldir = join(MACHINE_CONGIFS['container_dir'], exp_name, MACHINE_CONGIFS['modeldir'])
    logdir = join(MACHINE_CONGIFS['container_dir'], exp_name, MACHINE_CONGIFS['logdir'])
    slurmdir = join(MACHINE_CONGIFS['container_dir'], exp_name, MACHINE_CONGIFS['slurmdir'])
    outmap = {'outdir': outdir, 'modeldir':modeldir, 'logdir':logdir, 'slurmdir':slurmdir}
    mkdirs(outdir, modeldir, logdir, slurmdir)
    
    if exp_name.endswith('sa') or exp_name.endswith('l'):
        ac_modeldir = join(MACHINE_CONGIFS['container_dir'], exp_name, MACHINE_CONGIFS['ac_modeldir'])
        gan_modeldir = join(MACHINE_CONGIFS['container_dir'], exp_name, MACHINE_CONGIFS['gan_modeldir'])
        outmap.update({'ac_modeldir':ac_modeldir, 'gan_modeldir':gan_modeldir})
        mkdirs(ac_modeldir, gan_modeldir)
        
    return outmap


def parse(machine_name, exp_name):
    opt = importlib.import_module('conf.'+exp_name).get()

    opt['dataset']['root'] = MACHINE_CONGIFS[machine_name][opt['dataset']['name']]['root']

    opt['outdir'] = join(MACHINE_CONGIFS['container_dir'], exp_name, MACHINE_CONGIFS['outdir'])
    opt['modeldir'] = join(MACHINE_CONGIFS['container_dir'], exp_name, MACHINE_CONGIFS['modeldir'])
    opt['logdir'] = join(MACHINE_CONGIFS['container_dir'], exp_name, MACHINE_CONGIFS['logdir'])
    opt['slurmdir'] = join(MACHINE_CONGIFS['container_dir'], exp_name, MACHINE_CONGIFS['slurmdir'])
    if exp_name.endswith('sa') or exp_name.endswith('l'):
        opt['ac_modeldir'] = join(MACHINE_CONGIFS['container_dir'], exp_name, MACHINE_CONGIFS['ac_modeldir'])
        opt['gan_modeldir'] = join(MACHINE_CONGIFS['container_dir'], exp_name, MACHINE_CONGIFS['gan_modeldir'])
    return opt
