import os
from os.path import join
import sys
import conf

def start_exp(cluster_name, *exps):
    for e in exps:
        # Create output directory for the experiment
        slurm_out_dir = conf.create_dirs(e)['slurmdir']

        if cluster_name.startswith('crcv'):
            gpu = 'pascal' if cluster_name == 'crcv' else cluster_name.split(':')[-1]
            job_file = create_crcv_slurm(e, slurm_out_dir, gpu)
            os.system("sbatch %s" %job_file)
        elif cluster_name == 'newton':
            job_file = create_newton_slurm(e, slurm_out_dir)
            os.system("sbatch %s" %job_file)
        elif cluster_name == 'desktop':
            cmd = _get_train_file_execution_script(cluster_name, e)
            os.system(cmd)
        else:
            raise Exception('Unknown machine name')

def _get_train_file_execution_script(machine_name, exp_name):
    if exp_name.endswith('sa'):
        cmd = 'python train_srac.py {} {}'.format(machine_name, exp_name)
    elif exp_name.endswith('a'):
        cmd = 'python train_ac.py {} {}'.format(machine_name, exp_name)
    elif exp_name.endswith('p'):
        cmd = 'python train_progressive.py {} {}'.format(machine_name, exp_name)
    elif exp_name.endswith('l'):
        cmd = 'python train_loc.py {} {}'.format(machine_name, exp_name)
    else:
        cmd = 'python train_sr.py {} {}'.format(machine_name, exp_name)
    return cmd

def create_crcv_slurm(exp_name, slurmdir, gpu='pascal'):
    job_file = join(slurmdir, "%s.slurm" %exp_name)
    with open(job_file, 'w') as sf:
        sf.writelines("#!/bin/bash\n")
        sf.writelines("#SBATCH --job-name=%s\n" % exp_name)
        sf.writelines('#SBATCH -o "{slurm}/o_{exp}-%j.out"\n'.format(slurm=slurmdir, exp=exp_name))
        sf.writelines('#SBATCH --gres=gpu:{gpu}:1\n'.format(gpu=gpu))
        sf.writelines("#SBATCH -p gpu\n")
        sf.writelines("#SBATCH -c 4\n\n")

        sf.writelines('module load anaconda3 cuda/10.0 cudnn/10-74\n')
        sf.writelines('source activate pytorch_v1\n')

        sf.writelines(_get_train_file_execution_script('crcv', exp_name))
    return job_file


def create_newton_slurm(exp_name, slurmdir):
    job_file = join(slurmdir, "%s.slurm" %exp_name)
    with open(job_file, 'w') as sf:
        sf.writelines("#!/bin/bash\n")
        sf.writelines("#SBATCH --job-name=%s\n" % exp_name)
        sf.writelines('#SBATCH -o "{slurm}/o_{exp}-%j.out"\n'.format(slurm=slurmdir, exp=exp_name))

        sf.writelines('#SBATCH --nodes=1\n')
        sf.writelines('#SBATCH --cpus-per-task=4\n')
        sf.writelines('#SBATCH --ntasks=1\n')
        sf.writelines('#SBATCH --gres=gpu:1\n')
        sf.writelines('#SBATCH --time=48:00:00\n')

        sf.writelines(_get_train_file_execution_script('newton', exp_name))
    return job_file

def main():
    if len(sys.argv) < 3:
        print('Plese see the usage')
        print('#Usage:')
        print('starter.py <cluster_name>[:<gpu>] [<exp_name>,...]')
        print('    exp_name: <integer>[a|sa|p|]')
        print('        w/o postfix : train_sr.py')
        print('                  a : train_ac.py')
        print('                 sa : train_srac.py')
        print('                  p : train_progressive.py')
        print('    machine_name: [crcv|newton|desktop]')
        print('             gpu: [pascal | kepler | turing | volta] only for crcv')
    else:
        cluster_name = sys.argv[1]
        if cluster_name not in ['crcv','crcv:volta','crcv:pascal','crcv:kepler', 'crcv:turing','newton', 'desktop']: 
            raise Exception('Unknown cluster name')
        start_exp(cluster_name, *sys.argv[2:])

if __name__ == "__main__":
    main()
