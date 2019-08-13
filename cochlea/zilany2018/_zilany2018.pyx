# Copyright 2013-2014 Joerg Encke, Marek Rudnicki
# This file is part of cochlea.

# cochlea is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# cochlea is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with cochlea.  If not, see <http://www.gnu.org/licenses/>.


from __future__ import division, print_function, absolute_import

import numpy as np
from libc.stdlib cimport malloc
from . import util
import scipy.signal as dsp

cimport numpy as np

cdef extern from "stdlib.h":
    void *memcpy(void *str1, void *str2, size_t n)


cdef extern from "model_IHC_BEZ2018.h":
    void IHCAN(
        double *px,
        double cf,
        int nrep,
        double tdres,
        int totalstim,
        double cohc,
        double cihc,
        int species,
        double *ihcout
    )

cdef extern from "model_Synapse_BEZ2018.h":
    void SingleAN(double *px,
                  double cf,
                  int nrep,
                  double tdres,
                  int totalstim,
                  double noiseType,
                  double implnt,
                  double spont,
                  double tabs,
                  double trel,
                  double *meanrate,
                  double *varrate,
                  double *psth,
                  double *synout,
                  double *trd_vector,
                  double *trel_vector
    )


cdef extern from "Python.h":
    ctypedef int Py_intptr_t


cdef extern from "numpy/arrayobject.h":
    ctypedef Py_intptr_t npy_intp
    object PyArray_SimpleNewFromData(
        int nd,
        npy_intp* dims,
        int typenum,
        void* data
    )

np.import_array()

def run_ihc(
        np.ndarray[np.float64_t, ndim=1] signal,
        double cf,
        double fs,
        species='cat',
        double cohc=1.,
        double cihc=1.
):
    """Run middle ear filter, BM filters and IHC model.

    Parameters
    ----------
    signal : array_like
        Output of the middle ear filter in Pascal.
    cf : float
        Characteristic frequency in Hz.
    fs : float
        Sampling frequency in Hz.
    species : {'cat', 'human', 'human_glasberg1990'}
        Species.
    cihc, cohc : float
        Degeneration parameters for IHC and OHC cells.

    Returns
    -------
    array_like
        IHC receptor potential.

    """
    if species == 'cat':
        assert (cf > 124.9) and (cf < 40e3), "Wrong CF: 125 <= cf < 40e3, CF = %s"%str(cf)
    elif 'human' in species:
        assert (cf > 124.9) and (cf < 20001.), "Wrong CF: 125 <= cf <= 20e3, CF = %s"%str(cf)

    assert (fs >= 100e3) and (fs <= 500e3), "Wrong Fs: 100e3 <= fs <= 500e3"
    assert (cohc >= 0) and (cohc <= 1), "0 <= cohc <= 1"
    assert (cihc >= 0) and (cihc <= 1), "0 <= cihc <= 1"


    species_map = {
        'cat': 1,
        'human': 2,
        'human_glasberg1990': 3,
    }

    # Input sound
    if not signal.flags['C_CONTIGUOUS']:
        signal = signal.copy(order='C')
    cdef double *signal_data = <double *>np.PyArray_DATA(signal)

    # Output IHC voltage
    ihcout = np.zeros( len(signal) )
    cdef double *ihcout_data = <double *>np.PyArray_DATA(ihcout)


    IHCAN(
        signal_data,
        cf,
        1,
        1.0/fs,
        len(signal),
        cohc,
        cihc,
        species_map[species],
        ihcout_data
    )

    return ihcout

def run_single_an(
        np.ndarray[np.float64_t, ndim=1] vihc,
        double fs,
        double cf,
        anf_type='hsr',
        powerlaw='actual',
        ffGn=True,
        return_details=False):

    # Nr of repititons if larger then 1 then the output is the psth instead of a spike train
    nrep = 1

    # Decide if variable or frozen fGn
    if ffGn:
        noise_type = 1.
    else:
        noise_type = 0.

    powerlaw_map = {
        'actual': 1,
        'approximate': 0
    }

    # has to be re implemented in original its a random value drawn
    # out of a normal distribution with the following parameters mean
    # = (0.1, 4, 70), std = (0.1, 4, 70) bound within ([1e-3, 0.2],
    # [0.2, 18], [18, 180]) for LSR, MSR and HSR respectively
    spont = {
        'lsr': 0.1,
        'msr': 4.0,
        'hsr': 100.0,
    }

    # In original implementation the absolute and relative refreactory
    # period is randomly chosen to be within:
    # tabsmax = 1.5*461e-6;
    # tabsmin = 1.5*139e-6;
    # trelmax = 894e-6;
    # trelmin = 131e-6;
    tabs = (1.5*461e-6 + 1.5*139e-6) / 2
    trel = (894e-6 + 131e-6) / 2

    #  Input data
    cdef double *vihc_ptr = <double *>np.PyArray_DATA(vihc)

    # Output data
    mean_rate = np.zeros_like(vihc)
    cdef double *mean_rate_ptr = <double *>np.PyArray_DATA(mean_rate)
    var_rate = np.zeros_like(vihc)
    cdef double *var_rate_ptr = <double *>np.PyArray_DATA(var_rate)
    psth = np.zeros_like(vihc)
    cdef double *psth_ptr = <double *>np.PyArray_DATA(psth)
    synout = np.zeros_like(vihc)
    cdef double *synout_ptr = <double *>np.PyArray_DATA(synout)
    trd_vector = np.zeros_like(vihc)
    cdef double *trd_vector_ptr = <double *>np.PyArray_DATA(trd_vector)
    trel_vector = np.zeros_like(vihc)
    cdef double *trel_vector_ptr = <double *>np.PyArray_DATA(trel_vector)

    SingleAN(
        vihc_ptr,               # input ihc
        cf,                     # center frequency
        nrep,                   # nr of reps
        1.0 / fs,               # time resolutino
        len(vihc),              # length of stimulus
        noise_type,             # Noise type
        powerlaw_map[powerlaw], # aproximate or actual power law estimation
        spont[anf_type],        # spontanious rate
        tabs,                   # Absolute refractory period
        trel,                   # Relative refractory period
        mean_rate_ptr,          # Analytical estimate of mean firing rate
        var_rate_ptr,           # Analytical estimate of variance in firing rate
        psth_ptr,               # spiketrain
        synout_ptr,             # synapse output rate in 1/s for each time bin
        trd_vector_ptr,         # mean redocking time in s
        trel_vector_ptr         # mean realtive refractory perido in s for each time bin
    )


    detail_dict = {'mean rate' : mean_rate,
                   'var rate' : var_rate,
                   'synout' : synout,
                   'trd vector' : trd_vector,
                   'trel vector' : trel_vector}

    # if return_details:
    #     return psth, detail_dict
    # else:
    return psth

cdef public double* decimate(
    int k,
    double *signal,
    int q
):
    """Decimate a signal

    k: number of samples in signal
    signal: pointer to the signal
    q: decimation factor

    This implementation was inspired by scipy.signal.decimate.

    """
    # signal_arr will not own the data, signal's array has to be freed
    # after return from this function
    signal_arr = PyArray_SimpleNewFromData(
        1,                      # nd
        [k],                    # dims
        np.NPY_DOUBLE,          # typenum
        <void *>signal          # data
    )


    # resampled = dsp.resample(
    #     signal_arr,
    #     len(signal_arr) // q
    # )


    b = dsp.firwin(q+1, 1./q, window='hamming')
    a = [1.]

    filtered = dsp.filtfilt(
        b=b,
        a=a,
        x=signal_arr
    )

    resampled = filtered[::q]

    if not resampled.flags['C_CONTIGUOUS']:
        resampled = resampled.copy(order='C')

    # Copy data to output array
    cdef double *resampled_ptr = <double *>np.PyArray_DATA(resampled)
    cdef double *out_ptr = <double *>malloc(len(resampled)*sizeof(double))
    memcpy(out_ptr, resampled_ptr, len(resampled)*sizeof(double))

    return out_ptr


cdef public double* ffGn(int N, double tdres, double Hinput, double noiseType, double mu):
    """util.ffGn() wrapper"""

    a = util.ffGn(N, tdres, Hinput, noiseType, mu)

    if not a.flags['C_CONTIGUOUS']:
        a = a.copy(order='C')

    # Copy data to output array
    cdef double *ptr = <double *>np.PyArray_DATA(a)
    cdef double *out_ptr = <double *>malloc(len(a)*sizeof(double))
    memcpy(out_ptr, ptr, len(a)*sizeof(double))

    return out_ptr

cdef public double* generate_random_numbers(long length):
    arr = np.random.rand(length)

    if not arr.flags['C_CONTIGUOUS']:
        arr = arr.copy(order='C')

    cdef double *data_ptr = <double *>np.PyArray_DATA(arr)
    cdef double *out_ptr = <double *>malloc(length * sizeof(double))
    memcpy(out_ptr, data_ptr, length*sizeof(double))

    return out_ptr

cdef public double* sort_array(int nr_to_sort, double *in_ptr):

    in_array = PyArray_SimpleNewFromData(
        1,                      # nd
        [nr_to_sort],           # dims
        np.NPY_DOUBLE,          # typenum
        <void *>in_ptr          # data
    )

    sorted_array = np.sort(in_array)

    cdef double *sorted_ptr = <double *>np.PyArray_DATA(sorted_array)
    cdef double *out_ptr = <double *>malloc(nr_to_sort * sizeof(double))
    memcpy(out_ptr, sorted_ptr, nr_to_sort * sizeof(double))

    return out_ptr
