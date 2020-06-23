#!/usr/bin/env python3

import re
import hashlib
from pathlib import Path
from collections import namedtuple
from multiprocessing import Pool
import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

__version__ = '0.3.0'
__author__ = 'Kolja Glogowski'
__license__ = 'MIT'

# New-style random generator classes were introduced in NumPy version 1.17.0
_np_has_new_style_rngs = hasattr(np.random, 'Generator')

if _np_has_new_style_rngs:
    def init_rng_pcg64(seed=None):
        """Create a new-style RNG instance based on NumPy's Generator class"""
        return np.random.Generator(np.random.PCG64(seed))


def init_rng_legacy(seed=None):
    """Create a legacy RNG instance based on NumPy's RandomState class"""
    seed = seed % 2**32  # RandomState only supports unsigned 32-bit ints
    return np.random.RandomState(seed)  # pylint: disable=E1101


def create_random_data_file(fpath, size, rng=None, chunksize=65536,
                            overwrite=False, dryrun=False, sha1sum=False):
    """
    Generate a file containing random data.

    Parameters
    ----------
    fpath : str or pathlib.Path
        Path of the output file
    size : int
        Size of the output file
    rng : None or object
        Random number generator (RNG) used for creating the data. If `rng`
        is None, a randomly seeded RNG will be created.
    chunksize : int
        Chunksize used for generating/writing random data
    overwrite : bool
        Overwrite existing files. When `overwrite` is False (default), an
        exception will be raised, if the output file already exists.
    dryrun : bool
        Set to True, if no output should be written to file. If the target
        file already exists and if `overwrite` is False, an exception will
        be raised.
    sha1sum : bool
        Compute and return sha1sums of the generated file

    Returns
    -------
    result : pathlib.Path or (pathlib.Path, str)
        If sha1sum is set to False (default), this function returns the
        path of the file written. When sha1sum is set to True, the result
        is a tuple of path and SHA1 hexdigest, i.e. `(path, hexdigest)`
    """
    size = int(size)

    chunksize = int(chunksize)
    if chunksize <= 0:
        raise ValueError('Chunksize must be greater than 0')

    p = Path(fpath)
    if p.exists() and not overwrite:
        raise FileExistsError('File already exists: \'{}\''.format(p))

    if rng is None:
        rng = init_rng_legacy()

    def gen_random_chunk(size=size, rng=rng, chunksize=chunksize):
        while size > 0:
            n = min(size, chunksize)
            yield rng.bytes(n)
            size -= n

    if sha1sum:
        h = hashlib.sha1()

    if dryrun:
        for chunk in gen_random_chunk():
            if sha1sum:
                h.update(chunk)
    else:
        with p.open(mode='wb', buffering=chunksize) as f:
            for chunk in gen_random_chunk():
                if sha1sum:
                    h.update(chunk)
                f.write(chunk)

    if sha1sum:
        return p, h.hexdigest()
    else:
        return p


TaskArguments = namedtuple('TaskArguments', [
    'fpath', 'size', 'rng', 'chunksize', 'overwrite', 'dryrun', 'sha1sum'])


def _task_create_random_data_file(task_args):
    """
    Helper function that calls `create_random_data_file()`.

    This function is needed for executing the wrapped function using the
    `multiprocessing.Pool.imap()` method, which has no "starmap" equivalent,
    and therefore only supports task functions with single arguments.

    This function must be defined at the module's root level in order to
    be picklable, which is required when using `multiprocessing`.

    Parameters
    ----------
    task_args : TaskArguments
        Named tuple containing all task arguments

    Returns
    -------
    result : pathlib.Path
        Path of the created file
    """
    return create_random_data_file(
        fpath=task_args.fpath,
        size=task_args.size,
        rng=task_args.rng,
        chunksize=task_args.chunksize,
        overwrite=task_args.overwrite,
        dryrun=task_args.dryrun,
        sha1sum=task_args.sha1sum)


class RandomFileGenerator:
    def __init__(self, filesize, num_files, num_subdirs=0, seed=None,
                 rng='legacy'):
        """
        Random file generator.

        Parameters
        ----------
        filesize : int
            Size of each generated file
        num_files : int
            Number of files per (sub-)directory
        num_subdirs : int
            Number of sub-directories
        seed : int or None
            Initial random seed used for generating file data. If `seed` is
            None, a randomly picked value will be used.
        rng : 'legacy' or 'pcg64'
            Name of random number generator that is used for generating file
            data. The 'legacy' RNG (default) is using the class
            `numpy.random.RandomState` for generating random data, while the
            'pcg64' is using the new-style class `numpy.random.Generator` and
            the bit-generator `numpy.random.PCG64`, which were both introduced
            in NumPy version 1.17.0.
        """
        self._filesize = int(filesize)
        self._num_files = int(num_files)
        self._num_subdirs = int(num_subdirs)

        self._user_seed = seed
        self._initial_seed = self._compute_initial_seed()

        self._rng_name = rng
        if rng == 'legacy':
            self._rng_factory = init_rng_legacy
        elif rng == 'pcg64':
            self._rng_factory = init_rng_pcg64
        else:
            raise ValueError('Unknown RNG: {}'.format(rng))

    def __repr__(self):
        return ('{}(filesize={!r}, num_files={!r}, num_subdirs={!r}, '
                'seed={!r}, rng={!r})').format(
            self.__class__.__name__,
            self._filesize,
            self._num_files,
            self._num_subdirs,
            self._user_seed,
            self._rng_name)

    @property
    def filesize(self):
        """Size of each individual file"""
        return self._filesize

    @property
    def num_files(self):
        """Number of files per (sub-)directory"""
        return self._num_files

    @property
    def num_subdirs(self):
        """Number of sub-directories"""
        return self._num_subdirs

    @property
    def total_num_files(self):
        """Total number of files"""
        # The total number of files is (num_dirs * num_files_per_dir). If
        # num_subdirs is 0, the total number of directories is still 1.
        num_dirs = self._num_subdirs if self._num_subdirs > 0 else 1
        return num_dirs*self._num_files

    def dir_paths(self, root_dir):
        """Returns an iterator for directory paths used to create files"""
        root_path = Path(root_dir)
        if self._num_subdirs == 0:
            yield root_path
        else:
            ndigits = int(np.log10(self._num_subdirs)) + 1
            for i in range(self._num_subdirs):
                yield root_path / 'dir{:0{width}}'.format(i, width=ndigits)

    def file_paths(self, root_dir):
        """Returns an iterator for paths of the files to be created"""
        for dpath in self.dir_paths(root_dir):
            ndigits = int(np.log10(self._num_files)) + 1
            for i in range(self._num_files):
                yield dpath / 'file{:0{width}}'.format(i, width=ndigits)

    def _create_directories(self, root_dir, exist_ok=False):
        """Create all output directories"""
        Path(root_dir).mkdir(exist_ok=exist_ok)
        if self._num_subdirs != 0:
            for dpath in self.dir_paths(root_dir):
                dpath.mkdir(exist_ok=exist_ok)

    def _compute_initial_seed(self):
        """Compute initial random seed from characteristic job parameters"""
        if self._user_seed is not None:
            data = b'%d,%d,%d,%d' % (
                self._filesize,
                self._num_files,
                self._num_subdirs,
                self._user_seed)
            hexdigest = hashlib.sha1(data).hexdigest()
            return int(hexdigest[:16], base=16)
        else:
            # No seed was specified, so we just use a random integer as
            # initial seed for generating random data.
            return np.random.randint(2**63)

    def _get_imap_chunksize(self):
        """
        Try to figure out a reasonable `imap_chunksize` value.

        This function returns an `imap_chunksize` value that takes the
        filesize of the generated files into account. The current
        implementation returns a value of up to 100 for very small files,
        and continuously drops down to 1 for larger files.
        """
        # Don't bother, if we only generate a small number of files
        if self.total_num_files < 1000:
            return 1

        if self._filesize <= 2**10:
            # filesize <= 1 KiB
            return 100
        elif self._filesize >= 2**20:
            # filesize >= 1 MiB
            return 1
        else:
            # For 1 MiB > filesize > 1 KiB:
            # Increase value exponentially from 1 to 100.
            return int(10**((20 - np.log2(self._filesize))/5))

    def run(self, out_dir, num_workers=1, overwrite=False, dryrun=False,
            chunksize=65536, imap_chunksize=None, log=None, progress=False):
        """
        Generate random data files.

        This method provides a high-level interface that includes an
        optional progress bar and log output.

        Parameters
        ----------
        out_dir : str or pathlib.Path
            Output directory
        num_workers : int
            Number of parallel workers processes
        overwrite : bool
            Set to True to replace existing files. When set to False an
            exception will be raised, if a file or directory already exists.
        dryrun : bool
            Run the job without writing any file or directory. Exceptions
            are still raised, if overwrite is False and any file or
            directory already exists.
        chunksize : int
            Chunksize used for generating/writing random data
        imap_chunksize : int or None
            Chunksize used when calling `multiprocessing.Pool.imap()`.
            This parameter can be increased to improve performance when
            creating a large number of small files. When set to None
            (default), a value is choosen automatically by taking into
            account the file sizes and the total number of files.
        log : None, 'path', 'sha1sum'
            Control logging of completed file paths and hashsums. When set
            to None (default) no log output is printed. When `log` is set
            to 'path' the paths of completed files are printed. When set to
            'sha1sum' the path and SHA1 hexdigest of completed files are
            printed.
        progress : bool
            If set to True, a progress bar is shown
        """
        if log is None:
            verbose = False
            sha1sum = False
        elif log == 'path':
            verbose = True
            sha1sum = False
        elif log == 'sha1sum':
            verbose = True
            sha1sum = True
        else:
            raise ValueError('Unknown log mode: {}'.format(log))

        task_iter = self.run_iter(
            out_dir,
            num_workers=num_workers,
            overwrite=overwrite,
            dryrun=dryrun,
            chunksize=chunksize,
            imap_chunksize=imap_chunksize,
            sha1sum=sha1sum)

        if progress:
            with tqdm(total=self.total_num_files, unit='files') as pbar:
                for res in task_iter:
                    if verbose:
                        line = ('{!s}  {!s}'.format(res[1], res[0])
                                if sha1sum else str(res))
                        pbar.write(line)
                    pbar.update(1)
        else:
            if verbose:
                for res in task_iter:
                    line = ('{!s}  {!s}'.format(res[1], res[0])
                            if sha1sum else str(res))
                    print(line)
            else:
                for res in task_iter:
                    pass

    def run_iter(self, out_dir, num_workers=1, overwrite=False, dryrun=False,
                 chunksize=65536, imap_chunksize=None, sha1sum=False):
        """
        Generate random data files.

        This method returns an iterator over the paths (and optionally SHA1
        sums) of the generated files.

        Parameters
        ----------
        out_dir : str or pathlib.Path
            Output directory
        num_workers : int
            Number of parallel workers processes
        overwrite : bool
            Set to True to replace existing files. When set to False an
            exception will be raised, if a file or directory already exists.
        dryrun : bool
            Run the job without writing any file or directory. Exceptions
            are still raised, if overwrite is False and any file or
            directory already exists.
        chunksize : int
            Chunksize used for generating/writing random data
        imap_chunksize : int or None
            Chunksize used when calling `multiprocessing.Pool.imap()`.
            This parameter can be increased to improve performance when
            creating a large number of small files. When set to None
            (default), a value is choosen automatically by taking into
            account the file sizes and the total number of files.
        sha1sum : bool
            Compute and return sha1sums for each computed file

        Returns
        -------
        iter : iterator
            Iterator over the paths (and optionally sha1sums) of created
            files. When `sha1sum` is set to False (default), the result is an
            iterator over `pathlib.Path` objects. If `sha1sum` is set to
            True, the iterator elements are `(path, sha1sum)` tuples.
        """
        if imap_chunksize is None:
            imap_chunksize = self._get_imap_chunksize()

        # Generate sequence of seeded RNGs
        rng_iter = (
            self._rng_factory(seed) for seed in range(
                self._initial_seed,
                self._initial_seed + self.total_num_files))

        # Generate sequence of task arguments
        task_args_iter = (
            TaskArguments(
                fpath=fpath,
                size=self._filesize,
                rng=rng,
                chunksize=chunksize,
                overwrite=overwrite,
                dryrun=dryrun,
                sha1sum=sha1sum)
            for fpath, rng in zip(self.file_paths(out_dir), rng_iter))

        if not dryrun:
            self._create_directories(out_dir, exist_ok=overwrite)

        with Pool(processes=num_workers) as pool:
            for result in pool.imap_unordered(
                    _task_create_random_data_file, task_args_iter,
                    chunksize=imap_chunksize):
                yield result


def _parse_number_expr(expr, vmin=None, vmax=None):
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
        '--rng', choices=['legacy', 'pcg64'], default='legacy',
        help='Random number generator (default: legacy)')
    parser.add_argument(
        '-N', '--dry-run', dest='dryrun', action='store_true',
        help='Do not write anything to to disk')
    parser.add_argument(
        '-f', '--overwrite', action='store_true',
        help='Overwrite existing files')
    parser.add_argument(
        '-P', '--progress', action='store_true',
        help='Display progress bar')
    parser.add_argument(
        '-v', '--verbose', action='count',
        help='Print verbose output. Can be used twice to compute and show '
             'SHA1 sums of the generated files')
    parser.add_argument('--version', action='version', version=__version__)
    parser.add_argument(
        dest='out_dir', metavar='DIR', nargs=1, help='Output directory')
    args = parser.parse_args()

    if args.progress and tqdm is None:
        raise RuntimeError(
            'Cannot show progress bar. Install the tqdm package to use '
            'this feature.')

    # Parse/evaluate number expression
    filesize = _parse_number_expr(args.filesize, vmin=0)
    num_files_per_dir = _parse_number_expr(args.num_files_per_dir, vmin=1)
    num_subdirs = _parse_number_expr(args.num_subdirs, vmin=0)
    chunksize = _parse_number_expr(args.chunksize, vmin=1)

    # Log output verbosity levels
    if args.verbose is None:
        logmode = None
    elif args.verbose == 1:
        logmode = 'path'
    else:
        logmode = 'sha1sum'

    # Init random file generator
    rnd_file_gen = RandomFileGenerator(
        filesize=filesize,
        num_files=num_files_per_dir,
        num_subdirs=num_subdirs,
        seed=args.seed,
        rng=args.rng)

    # Generate files
    rnd_file_gen.run(
        args.out_dir[0],
        num_workers=args.num_workers,
        overwrite=args.overwrite,
        dryrun=args.dryrun,
        chunksize=chunksize,
        imap_chunksize=None,
        log=logmode,
        progress=args.progress)
