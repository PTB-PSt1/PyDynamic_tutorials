# -*- coding: utf-8 -*-
"""
This module contains methods for the estimation of the
regularization bias/error and its uncertainty contribution.
"""
import numpy as np
import scipy.signal as dsp

def calculate_freq_points(fH, S, candidates):
    """ Calculate wavelet center frequencies from list of candidates,
    such that the center frequencies correspond to actual frequencies of
    the measurand's spectrum local maxima

    Parameters
    ----------
        fH : np.ndarry of shape (M,)
             frequencies for spectra H and X
        S : np.ndarray of shape (2M,)
            spectrum of reference signal as [real, imag]
        candidates : list or np.ndarray
                     candidate frequencies for center frequencies

    Returns
    -------
        frequencies : np.ndarray
                      wavelet center frequencies

    """
    N = len(S)//2
    Xh = S[:N] + 1j*S[N:]

    # Largest local maxima of np.abs(Xh)
    inds = dsp.argrelmax(np.abs(Xh))[0]
    inds2 = inds[np.argsort(np.abs(Xh[inds]))[::-1]]
    f_local = fH[inds2[:50]]

    # identification of local max close to candidates
    frequencies = np.zeros_like(candidates)
    for k in range(len(frequencies)):
        indf = np.argmin(np.abs(f_local - candidates[k]))
        frequencies[k] = f_local[indf]

    return frequencies

class Gaussian_Wavelet:
    """

    """
    def __init__(self, A = 1.0, sigma = None, freq = None, verbose = True):
        self.A = A
        self.freq = freq
        if sigma is None and isinstance(freq, float):
            self.sigma = self.freq2scale(freq)
        else:
            self.sigma = sigma
        if verbose:
            print("characterising frequency: %g MHz" % (self.freq * 1e-6), end="")
            print(" corresponding sigma: %g" % (self.sigma), end="")
            print(" and amplitude %g" % self.A)

        self.spectrum = lambda f: self.A * np.exp(0.5) * self.sigma * (2 * np.pi * f) * \
                         np.exp(-0.5 * self.sigma ** 2 * (2 * np.pi * f) ** 2)

    def freq2scale(self, f):
        """Calculation of wavelet scale from center frequency

        Parameters
        ----------
        f : float or np.ndarray
            frequency value(s)
        Returns
        -------
            float or np.ndarray, scale value(s)
        """
        return 1 / (2 * np.pi * f)


class Wavelet_Bound:
    """
    Upper bound function based on Gaussian wavelets in the frequency domain
    """
    def __init__(self):
        self.wavelets = []
        self.bound_function = None

    def add_wavelet(self, A = 1.0, sigma = None, freq = None):
        """Add a new wavelet to the set of wavelets
            Parameters
            ----------
            A : float, amplitude scaling
            sigma : float, wavelet scaling factor
        """
        Wavelet = Gaussian_Wavelet(A=A, sigma=sigma, freq=freq)
        self.wavelets.append(Wavelet)
        self.update_bound_function()

    def update_bound_function(self):
        """ Re-evaluate the bound function as sum of the wavelet spectra functions
        This is necessary whenever a new wavelet has been added.
        """
        self.bound_function = lambda f: sum([wavelet.spectrum(f) for wavelet in self.wavelets])

    def calc_bound_value(self, f, G, H, Fs):
        """
        Calculation of the upper bound on the dynamic error in the time domain
        from the upper bound in the frequency domain.

        Parameters
        ----------
            f : np.ndarray of shape (M,)
                frequencies
            G : np.ndarray of shape (M,)
                frequency response of applied deconvolution filter at frequencies f
            H : np.ndarray of shape (M,) or lambda function with argument f
                frequency response of measurement system
            Fs: float
                sampling frequency

        Returns
        -------
            float
            upper bound of dynamic error
        """

        freq_devs = self.bound_function(f) * np.abs(G * H - 1)
        return 1 / (2 * np.pi) * np.trapz(freq_devs, f)


def calculate_bound(fS, S, freq_points, f, scaling = 1.0):
    """Calculate the bound function from the measured signal and the measuring instrument's frequency response

    Parameters
    ----------
        fS : np.ndarray of shape (M,)
            frequencies at which the reference signal spectrum is defined
        S : np.ndarray of shape (M,) and dtype complex
            frequency domain values of reference signal for bound calculation
        freq_points : list
            frequencies at which the Gaussian wavelets are centered
        f : np.ndarray of shape (N,)
            equidistant frequencies from 0 to Nyquist (Fs/2)
        scaling : float
            scaling factor for bound function

    Returns
    -------
        B : np.ndarray of shape (N,)
            spectrum of bound
        Bound : instance of class Wavelet_Bound
            resulting bound as a class object
    """

    def get_amplitude(Amps, freqs, fv):
        ind1 = np.nonzero(freqs <= fv)[0][-1]
        ind2 = ind1 + 1
        f1, A1 = freqs[ind1], Amps[ind1]
        f2, A2 = freqs[ind2], Amps[ind2]
        return (A2 - A1) / (f2 - f1) * (fv - f1) + A1

    amplitude_vals = [get_amplitude(np.abs(S), fS, ff) for ff in freq_points]
    Bound = Wavelet_Bound()
    for A, freq in zip(amplitude_vals, freq_points):
        Bound.add_wavelet(A, freq=freq)

    Fs = f[-1] * 2
    B = scaling * Bound.bound_function(f) / Fs

    return B, Bound

