"""
This script combines output from multiple runs (e.g., where one run was restarted in multiple different directories).

    Usage:
        merge_directories.py <dirs> <dirs>...

"""
import os
import glob
import shutil

import numpy as np
from mpi4py import MPI
from docopt import docopt
args = docopt(__doc__)

if MPI.COMM_WORLD.rank == 0:
    dirs = args['<dirs>']
    for d in dirs:
        print(d)
        if 'stitched' in d:
            dirs.remove(d)


    base_dir = dirs[0]
    out_dir = base_dir
    #remove '/' from end of out_dir
    while out_dir[-1] == '/':
        out_dir = out_dir[:-1]
    out_dir += '_stitched/'
    print('out_dir {}'.format(out_dir))
    if not os.path.exists('{:s}'.format(out_dir)):
        print('creating out_dir {}'.format(out_dir))
        os.mkdir('{:s}'.format(out_dir))


    for sub_dir in ['profiles', 'scalars', 'slices', 'checkpoint']:
        curr_out_dir = '{}/{}/'.format(out_dir, sub_dir)
        if not os.path.exists('{:s}'.format(curr_out_dir)):
            print('creating out_dir {}'.format(curr_out_dir))
            os.mkdir('{:s}'.format(curr_out_dir))

        global_file_nums = []
        for root_dir in dirs:
            files = glob.glob('{}/{}/{}_s*.h5'.format(root_dir, sub_dir, sub_dir))
            if len(files) == 0:
                print('no files in {}/{}, continuing'.format(root_dir, sub_dir))
                continue
            sorted_files = sorted(files, key= lambda f : int(f.split('.h5')[0].split('_s')[-1]))
            file_num_in = np.array([int(f.split('.h5')[0].split('_s')[-1]) for f in sorted_files])
            file_num_out = np.copy(file_num_in)
            if len(global_file_nums) != 0:
                print(global_file_nums)
                file_num_out += global_file_nums[-1][-1]
#            print(file_num_out)
            global_file_nums.append(file_num_out)
            
            sorted_files_out = []
            for n in file_num_out:
                sorted_files_out.append('{}/{}_s{}.h5'.format(curr_out_dir, sub_dir, n))

            for fi, fo in zip(sorted_files, sorted_files_out):
                print('{} -> {}'.format(fi, fo))
                shutil.copyfile(fi, fo)
