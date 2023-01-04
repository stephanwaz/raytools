# -*- coding: utf-8 -*-

# Copyright (c) 2019 Stephen Wasilewski
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================

"""progress bar"""
import shutil
from datetime import datetime

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures import wait, FIRST_COMPLETED
from multiprocessing import get_context

from tqdm import tqdm
from raytools import io


class TStqdm(tqdm):

    def __init__(self, instance=None, tz=None, workers=False, position=0,
                 desc=None, ncols=100, cap=None, **kwargs):
        if str(workers).lower() in ('thread', 't', 'threads'):
            pool = ThreadPoolExecutor()
        elif workers:
            context = get_context('fork')
            nproc = io.get_nproc(cap)
            pool = ProcessPoolExecutor(nproc, mp_context=context)
        else:
            pool = None
        self._instance = instance
        self.loglevel = position
        tf = "%H:%M:%S"
        self.ts = datetime.now(tz=tz).strftime(tf)
        self.pool = pool
        self.wait = wait
        self.FIRST_COMPLETED = FIRST_COMPLETED
        if pool is None:
            self.nworkers = 0
        else:
            self.nworkers = pool._max_workers
        ncols = min(ncols, shutil.get_terminal_size().columns)
        super().__init__(desc=self.ts_message(desc), position=position,
                         ncols=ncols, **kwargs)

    def ts_message(self, s):
        if self._instance is not None:
            p = type(self._instance).__name__
        else:
            p = ""
        if s is None:
            s = f"{p}"
        else:
            s = f"{p} {s}"
        s = f"{' | ' * self.loglevel} {s}"
        return s

    def write(self, s, file=None, end="\n", nolock=False):
        super().write(self.ts_message(s), file, end, nolock)

    def set_description(self, desc=None, refresh=True):
        super().set_description(desc=self.ts_message(desc), refresh=refresh)


def pool_call(func, args, *fixed_args, cap=None, expandarg=True,
              desc="processing", workers=True, pbar=True, **kwargs):
    """calls func for a sequence of arguments using a ProcessPool executor
    and a progress bar. result is equivalent to::

         result = []
         for arg in args:
             result.append(func(*args, *fixed_args, **kwargs))
         return result

    Parameters
    ----------
    func: callable
        the function to execute in parallel
    args: Sequence[Sequence]
        list of arguments (each item is expanded with '*' unless expandarg
        is false). first N args of func
    fixed_args: Sequence
        arguments passed to func that are the same for all calls (next N
        arguments  after args)
    cap: int, optional
        execution cap for ProcessPool
    expandarg: bool, optional
        expand args with '*' when calling func
    desc: str, optional
        label for progress bar
    workers: Union[bool, str], optional
        return threadpool ('t', 'threads', 'thread') or processpool (True)
    pbar: bool, optional
        display progress bar while executing
    kwargs:
        additional keyword arguments passed to func
    Returns
    -------
    sequence of results from func (order preserved)
    """
    results = []
    if not workers:
        result = []
        for arg in args:
            if expandarg:
                result.append(func(*arg, *fixed_args, **kwargs))
            else:
                result.append(func(arg, *fixed_args, **kwargs))
        return result
    with TStqdm(workers=workers, total=len(args), cap=cap,
                desc=desc, disable=not pbar) as pbar:
        exc = pbar.pool
        futures = []
        # submit asynchronous to process pool
        for arg in args:
            if expandarg:
                fu = exc.submit(func, *arg, *fixed_args, **kwargs)
            else:
                fu = exc.submit(func, arg, *fixed_args, **kwargs)
            futures.append(fu)
        # gather results (in order)
        for future in futures:
            results.append(future.result())
            pbar.update(1)
    return results
