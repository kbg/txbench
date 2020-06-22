#!/usr/bin/env python3

import re
import hashlib
from pathlib import Path
import numpy as np

__version__ = '0.2.0'
__author__ = 'Kolja Glogowski'
__license__ = 'MIT'


def create_dir_path_list(root_dir, num_subdirs=0):
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


def create_file_path_iter(dpath_list, num_files_per_dir):
    for dpath in dpath_list:
        ndigits = int(np.log10(num_files_per_dir)) + 1
        for i in range(num_files_per_dir):
            fpath = dpath / 'file{:0{width}}'.format(i, width=ndigits)
            yield fpath


def create_directories(root_dir, dpath_list, exist_ok=False):
    root_path = Path(root_dir)
    root_path.mkdir(exist_ok=exist_ok)
    if len(dpath_list) == 1 and dpath_list[0] == root_path:
        return
    for dpath in dpath_list:
        dpath.mkdir(exist_ok=exist_ok)


def init_rng_legacy(seed=None):
    """Create a legacy RNG instance based on NumPy's RandomState class"""
    seed = seed % 2**32  # RandomState only supports unsigned 32-bit ints
    return np.random.RandomState(seed)  # pylint: disable=E1101


# If available, use new-style random generator classes as a default (these
# classes were introduced in NumPy version 1.17.0).
_np_has_new_style_rngs = hasattr(np.random, 'Generator')
if _np_has_new_style_rngs:
    def init_rng_pcg64(seed=None):
        """Create a new-style RNG instance based on NumPy's Generator class"""
        return np.random.Generator(np.random.PCG64(seed))
    init_rng_default = init_rng_pcg64
else:
    init_rng_pcg64 = None
    init_rng_default = init_rng_legacy


def compute_initial_seed(filesize, num_files, num_subdirs, seed):
    """Compute initial random seed from characteristic job parameters"""
    data = b'%d,%d,%d,%d' % (filesize, num_files, num_subdirs, seed)
    hexdigest = hashlib.sha1(data).hexdigest()
    return int(hexdigest[:16], base=16)


def create_random_data_file(fpath, size, rng=None, chunksize=65536,
                            overwrite=False, dryrun=False):
    size = int(size)
    chunksize = int(chunksize)
    if chunksize <= 0:
        raise ValueError('Chunksize must be greater than 0')

    p = Path(fpath)
    if p.exists() and not overwrite:
        raise FileExistsError('File already exists: \'{}\''.format(p))

    if rng is None:
        rng = init_rng_default()

    if dryrun:
        while size > 0:
            n = min(size, chunksize)
            rng.bytes(n)
            size -= n
    else:
        with p.open(mode='wb', buffering=chunksize) as f:
            while size > 0:
                n = min(size, chunksize)
                f.write(rng.bytes(n))
                size -= n
    return p


def parse_number_expr(expr, vmin=None, vmax=None):
    if re.match(r'^[0-9/\+\*\-\(\)e]+$', expr):
        value = int(eval(expr))
    else:
        raise ValueError('Cannot parse argument: \'{}\''.format(expr))

    if vmin is not None and value < vmin:
        raise ValueError('Value is lower than {}: {}'.format(vmin, value))
    if vmax is not None and value > vmax:
        raise ValueError('Value is greater than {}: {}'.format(vmax, value))

    return value


if __name__ == '__main__':
    import argparse
    from multiprocessing import Pool

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-s', dest='filesize', metavar='FILESIZE', type=str, required=True,
        help='Size of generated files in bytes')
    parser.add_argument(
        '-n', dest='num_files_per_dir', metavar='NUM_FILES', type=str,
        required=True, help='Number of files per directory')
    parser.add_argument(
        '-d', dest='num_subdirs', metavar='NUM_DIRS', type=str, default='0',
        help='Number of subdirectories (default: 0)')
    parser.add_argument(
        '-c', dest='chunksize', type=str, default='65536',
        help='Chunk size used for writing files (default: 65536)')
    parser.add_argument(
        '-j', dest='num_workers', metavar='NUM_JOBS', type=int,
        default=1, help='Number of parallel jobs (default: 1)')
    parser.add_argument(
        '-S', '--seed', type=int,
        help='Initial seed for generating random data')
    parser.add_argument(
        '--rng', choices=['pcg64', 'legacy'],
        help=('Random number generator. If not explicitely specified, '
              '"pcg64" is used for newer NumPy versions (>= 1.7.0) and '
              '"legacy" is used for older versions (< 1.7.0).'))
    parser.add_argument(
        '--dry-run', dest='dryrun', action='store_true',
        help='Do not write anything to to disk')
    parser.add_argument(
        '-f', '--overwrite', action='store_true',
        help='Overwrite existing files')
    parser.add_argument(
        '-P', '--progress', action='store_true',
        help='Display progress bar')
    parser.add_argument(
        '-v', '--verbose', action='store_true',
        help='Print verbose output')
    parser.add_argument('--version', action='version', version=__version__)
    parser.add_argument(
        dest='out_dir', metavar='DIR', nargs=1, help='Output directory')
    args = parser.parse_args()

    if args.progress and tqdm is None:
        raise RuntimeError(
            'Cannot show progress bar. Install the tqdm package to use '
            'this feature.')

    filesize = parse_number_expr(args.filesize, vmin=0)
    num_files_per_dir = parse_number_expr(args.num_files_per_dir, vmin=1)
    num_subdirs = parse_number_expr(args.num_subdirs, vmin=0)
    chunksize = parse_number_expr(args.chunksize, vmin=1)

    # The total number of files is (num_dirs * num_files_per_dir). If
    # num_subdirs is 0, the total number of directories is still 1.
    total_num_files = num_files_per_dir*(
        num_subdirs if num_subdirs > 0 else 1)

    if args.seed is not None:
        # The initial random seed is computed from all characteristic
        # parameters, including the user-suplied seed.
        initial_seed = compute_initial_seed(
            filesize=filesize,
            num_files=num_files_per_dir,
            num_subdirs=num_subdirs,
            seed=args.seed)
    else:
        # No seed was specified, so we just use a random integer as initial
        # seed for generating random data.
        initial_seed = np.random.randint(2**63)

    if args.rng == 'pcg64':
        init_rng = init_rng_pcg64
    elif args.rng == 'legacy':
        init_rng = init_rng_legacy
    else:
        init_rng = init_rng_default

    out_dir = args.out_dir[0]
    dpath_list = create_dir_path_list(out_dir, num_subdirs)

    # Generate sequence of file paths
    fpath_iter = create_file_path_iter(dpath_list, num_files_per_dir)

    # Generate sequence of seeded RNGs
    rng_iter = (init_rng(seed) for seed in range(
        initial_seed, initial_seed + total_num_files))

    if not args.dryrun:
        create_directories(out_dir, dpath_list, exist_ok=args.overwrite)

    # We need to create a wrapper function that unpacks (fpath, rng)
    # tuples, because Pool.imap_unordered() only supports a single variable
    # argument and there is no starmap() equivalent for imap().
    def task_func(args, size=filesize, chunksize=chunksize,
                  overwrite=args.overwrite, dryrun=args.dryrun):
        fpath, rng = args
        return create_random_data_file(
            fpath=fpath, size=size, rng=rng, chunksize=chunksize,
            overwrite=overwrite, dryrun=dryrun)

    with Pool(processes=args.num_workers) as pool:
        if args.progress:
            with tqdm(total=total_num_files, unit='files') as pbar:
                for res in pool.imap_unordered(
                        task_func, zip(fpath_iter, rng_iter)):
                    if args.verbose:
                        pbar.write(str(res))
                    pbar.update(1)
        else:
            for res in pool.imap_unordered(
                    task_func, zip(fpath_iter, rng_iter)):
                if args.verbose:
                    print(str(res))
