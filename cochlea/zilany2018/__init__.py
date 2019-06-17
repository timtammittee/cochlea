from __future__ import division, print_function, absolute_import

import itertools
import numpy as np
import pandas as pd

from . import _zilany2018
from . util import calc_cfs
# from . zilany2014_rate import run_zilany2014_rate


def run_zilany2018(
        sound,
        fs,
        anf_num,
        cf,
        species,
        seed,
        cohc=1,
        cihc=1,
        powerlaw='approximate',
        ffGn=False
):
    """Run the inner ear model by [Bruce2018]_.

    This model is based on the original implementation provided by the
    authors.  The MEX specific code was replaced by Python code in C
    files.  We also compared the outputs of both implementation (see
    the `tests` directory for unit tests).


    Parameters
    ----------
    sound : array_like
        The input sound in Pa.
    fs : float
        Sampling frequency of the sound in Hz.
    anf_num : tuple
        The desired number of auditory nerve fibers per frequency
        channel (CF), (HSR#, MSR#, LSR#).  For example, (100, 75, 25)
        means that we want 100 HSR fibers, 75 MSR fibers and 25 LSR
        fibers per CF.
    cf : float or array_like or tuple
        The center frequency(s) of the simulated auditory nerve fibers.
        If float, then defines a single frequency channel.  If
        array_like (e.g. list or ndarray), then the frequencies are
        used.  If tuple, then must have exactly 3 elements (min_cf,
        max_cf, num_cf) and the frequencies are calculated using the
        Greenwood function.
    species : {'cat', 'human', 'human_glasberg1990'}
        Species.
    seed : int
        Random seed for the spike generator.
    cohc : float <0-1>, optional
        Degredation of the outer hair cells.
    cihc : float <0-1>, optional
        Degredation of the inner hair cells.
    powerlaw : {'approximate', 'actual'}, optional
        Defines which power law implementation should be used.
    ffGn : bool
        Enable/disable factorial Gaussian noise.


    Returns
    -------
    spike_trains
        Auditory nerve spike trains.


    References
    ----------
    If you are using results of this or modified version of the model
    in your research, please cite [Bruce2018]_.

    .. [Bruce2018] Bruce, I. C., Erfani, Y., & Zilany, M. S. (2018). A
    phenomenological model of the synapse between the inner hair cell
    and auditory nerve: implications of limited neurotransmitter release
    sites. Hearing Research, 360(),

    """
    assert np.max(sound) < 1000, "Signal should be given in Pa"
    assert sound.ndim == 1
    assert species in ('cat', 'human', 'human_glasberg1990')

    np.random.seed(seed)

    # TODO: use cochlea.greenwood() instead
    cfs = calc_cfs(cf, species)

    channel_args = [
        {
            'signal': sound,
            'cf': cf,
            'fs': fs,
            'cohc': cohc,
            'cihc': cihc,
            'anf_num': anf_num,
            'powerlaw': powerlaw,
            'seed': seed,
            'species': species,
            'ffGn': ffGn,
        }
        for cf in cfs
    ]


    ### Run model for each channel
    nested = map(
        _run_channel,
        channel_args
    )


    ### Unpack the results
    trains = itertools.chain(*nested)
    spike_trains = pd.DataFrame(list(trains))


    if isinstance(np.fft.fftpack._fft_cache, dict):
        np.fft.fftpack._fft_cache = {}

    return spike_trains

def _run_channel(args):

    fs = args['fs']
    cf = args['cf']
    signal = args['signal']
    cohc = args['cohc']
    cihc = args['cihc']
    powerlaw = args['powerlaw']
    seed = args['seed']
    anf_num = args['anf_num']
    species = args['species']
    ffGn = args['ffGn']


    ### Run BM, IHC
    vihc = _zilany2018.run_ihc(
        signal=signal,
        cf=cf,
        fs=fs,
        species=species,
        cohc=float(cohc),
        cihc=float(cihc)
    )


    duration = len(vihc) / fs
    anf_types = np.repeat(['hsr', 'msr', 'lsr'], anf_num)

    synout = {}
    trains = []
    time = 1./fs * np.arange(len(vihc))
    for anf_type in anf_types:
        spikes  = _zilany2018.run_single_an(vihc,
                                            fs,
                                            cf,
                                            anf_type,
                                            powerlaw,
                                            ffGn)

        spikes = time[np.where(spikes!=0)]

        trains.append({
            'spikes': spikes,
            'duration': duration,
            'cf': args['cf'],
            'type': anf_type
        })


    return trains
