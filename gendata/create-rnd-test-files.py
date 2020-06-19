#!/usr/bin/env python3

import re
from pathlib import Path
import numpy as np
import dask

__version__ = '0.1'
__author__ = 'Kolja Glogowski'
__license__ = 'MIT'


def create_dpath_list(root_dir, num_subdirs=0):
    root = Path(root_dir)
    res = []
    if num_subdirs == 0:
        res.append(root)
    else:
        ndigits = int(np.log10(num_subdirs)) + 1
        for i in range(num_subdirs):
            p = root / 'dir{:0{width}}'.format(i, width=ndigits)
            res.append(p)
    return res


def create_fpath_list(dpath_list, num_files_per_dir):
    res = []
    for dpath in dpath_list:
        ndigits = int(np.log10(num_files_per_dir)) + 1
        for i in range(num_files_per_dir):
            fpath = dpath / 'file{:0{width}}'.format(i, width=ndigits)
            res.append(fpath)
    return res


def create_directories(root_dir, dpath_list, exist_ok=False):
    root_path = Path(root_dir)
    root_path.mkdir(exist_ok=exist_ok)
    if len(dpath_list) == 1 and dpath_list[0] == root_path:
        return
    for dpath in dpath_list:
        dpath.mkdir(exist_ok=exist_ok)


def gen_rnd_data_file(fpath, size, chunksize=1048576, overwrite=False):
    size = int(size)
    chunksize = int(chunksize)
    if chunksize <= 0:
        raise ValueError('Chunksize must be greater than 0')
    p = Path(fpath)
    if p.exists() and not overwrite:
        raise FileExistsError('File already exists: \'{}\''.format(p))
    with p.open(mode='wb', buffering=chunksize) as f:
        while size > 0:
            n = min(size, chunksize)
            f.write(np.random.bytes(n))
            size -= n


def parse_number_expr(expr):
    if re.match(r'^[0-9/\+\*\-\(\)e]+$', expr):
        return int(eval(expr))
    else:
        raise RuntimeError('Cannot parse argument: \'{}\''.format(name, expr))


if __name__ == '__main__':
    import argparse
    from dask.diagnostics import ProgressBar

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-n', dest='filesize', metavar='FILESIZE', type=str, required=True,
        help='File sizes in bytes')
    parser.add_argument(
        '-f', dest='num_files_per_dir', metavar='NUM_FILES', type=str,
        required=True, help='Number of files per directory')
    parser.add_argument(
        '-d', dest='num_subdirs', metavar='NUM_DIRS', type=str, default='0',
        help='Number of subdirectories (default: 0)')
    parser.add_argument(
        '-c', dest='chunksize', metavar='CHUNKSIZE', type=int, default=1048576,
        help='Chunk size used for writing files (default: 1048576)')
    parser.add_argument(
        '--workers', dest='num_workers', metavar='N', type=int, default=1,
        help='Number of worker threads (default: 1)')
    parser.add_argument(
        '--overwrite', action='store_true', help='Overwrite existing files')
    parser.add_argument(
        dest='out_dir', metavar='DIR', nargs=1, help='Output directory')
    args = parser.parse_args()

    filesize = parse_number_expr(args.filesize)
    num_files_per_dir = parse_number_expr(args.num_files_per_dir)
    num_subdirs = parse_number_expr(args.num_subdirs)

    out_dir = args.out_dir[0]
    dpath_list = create_dpath_list(out_dir, num_subdirs)
    fpath_list = create_fpath_list(dpath_list, num_files_per_dir)
    create_directories(out_dir, dpath_list, exist_ok=args.overwrite)

    dask.config.set(num_workers=args.num_workers)
    job_list = [dask.delayed(gen_rnd_data_file)(
                    fpath,
                    size=filesize,
                    chunksize=args.chunksize,
                    overwrite=args.overwrite)
                for fpath in fpath_list]

    with ProgressBar():
        dask.compute(*job_list)
