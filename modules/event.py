#########1#########2#########3#########4#########5#########6#########7#########
"FRB pulse generation"
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve

class Radiometer():
    def __init__(
        self, 
        f_range=(4400, 5000),  # total frequency range
        n_chan=4,              # N of radio channels
        tstep=0.000245,        # time step
        n_tpoints=4080,        # N of points in the time series
        centering=True,        # pulse at the center
        l_bound=500,           # if not centered: the leftmost bound
        verbose=False          # some messages
    ):
        self.f_range = f_range
        self.n_chan = n_chan
        self.tstep = tstep
        self.n_tpoints = n_tpoints
        self.centering = centering
        self.verbose = verbose
        self.pulse = np.empty((self.n_chan, self.n_tpoints))
                
        # reference frequency
        self.ref_freq = f_range[0] + (f_range[1]-f_range[0])/2

        # centers of frequency bands
        self.delta_freq = (f_range[1] - f_range[0]) / self.n_chan
        self.f_centers = np.array(
            [f_range[0] + self.delta_freq*(i+0.5) for i in range(self.n_chan)])
        
        # Limits of gaussian center shift
        l_bound = n_tpoints/2 - l_bound
        if self.centering:
            self.max_shift = 0
        else:    
            self.max_shift = l_bound if  l_bound > 0 else 0
            
    def gaussian_profile(self, nt, width, t0=0.):
        """ Normalized Gaussian as a pulse
        """
        t = np.linspace(-nt//2, nt//2, nt)
        #power = 2 + 2 * np.random.binomial(1, 0.5)
        width = width / (2*np.sqrt(2*np.log(2)))  # gaussian sigma
        g = np.exp(-(t-t0)**2 / (2*width**2))

        if not np.all(g > 0):
            g += 1e-18

        return g / g.max()

    def get_delays(self, f_centers, dm):
        """ Delays per channel
            Higher frequency arrives first
        """
        return (
            1E4 / 2.41 * (1/f_centers**2 - 1/f_centers[-1]**2) * dm)

    def scat_profile(self, nt, tau_nu):
        """ Exponential scattering profile
        """
        t = np.linspace(0, nt, nt)
        prof = np.exp(-t/tau_nu)
        return prof / prof.max()
    
    def scintillation(self, f_centers, ref_freq):
        """ 
        Spectral scintillation across the band. Approximate effect as 
        a sinusoid with random phase and random frequency 
        """
        # Random "number" of patches (frequency of sinusoid peaks), float
        #second_exist = False
        #while not second_exist:  # at least 2 channels must be nonzero
        #nscint = np.exp(np.random.uniform(np.log(0.9), np.log(10)))
        nscint = np.exp(np.random.uniform(np.log(0.5), np.log(10)))
        if nscint < 1:
            return np.ones(len(f_centers))
                
        # Make location of peaks random
        scint_phi = np.random.rand()

        scint_amp = np.cos(
            2*np.pi*nscint*(f_centers/ref_freq)**2 + scint_phi)
        scint_amp[scint_amp <= 0] = 1E-18
        #    second_exist = sorted(scint_amp)[1] > 0.1
        
        return scint_amp 
    
    def pulse_profile(self, nt, width, f, tau, 
                      t0, dm, delta_freq, tsamp):
        """ Convolution of the gaussian and scattering profiles 
        into the final pulse shape at a frequency channel.
        ----------
        nt: int
            number of time points (record length)
        width: int 
            gaussian width [nt]
        f:  float 
            frequency [MHz]
        tau: int 
            scattering time at 1 GHz [nt]
        t0: float
            profile center [nt]
        dm: float
            dispersion measure [pc cm-3]
        delta_freq : float 
            freq channel width [MHz]
        tsamp : float 
            time step for nt [s]
        """
        gaus_prof = self.gaussian_profile(nt, width, t0=t0)

        # Scattering
        tau += 1e-18  # avoiding zero division in scat_profile()
        tau_nu = tau * (f / 1000.)**-4.  # power dependence on frequency: -4
        scat_prof = self.scat_profile(nt, tau_nu) 
        pulse_prof = fftconvolve(gaus_prof, scat_prof)[:nt]
        
        # Dispersion measure smearing
        tdm = 8.3e-6 * dm * delta_freq / (f*1e-3)**3
        tdm_samp = tdm / tsamp
        dm_smear_prof = np.ones(max(1, round(tdm_samp)))
        pulse_prof = fftconvolve(pulse_prof, dm_smear_prof, mode='same')
        
        # Intensity recalibration
        pulse_prof /= pulse_prof.max()
        pulse_prof *= (width/np.sqrt(width**2 + tau_nu**2 + tdm_samp**2))

        return pulse_prof

    def get_pulse(
        self,
        ampl=5,        # amplitude
        width=0.0005,  # width in seconds
        tau=0.004,     # scattering time at 1 GHz in seconds
        dm=50,         # dispersion measure in (pc cm−3)
        sp_ind=0,      # spectral index
        noise=False,   # add a Gaussian noise with sigma = 1 ampl
        scint=False    # scintillations (insert randomly in ~22% of events)
    ):
        """ Generate a pulse: nd.array [n_chan, n_tpoints]
        """
        delays = self.get_delays(self.f_centers, dm) / self.tstep
        if self.centering:
            t0 = -delays[0] / 2
        else:
            t0 = np.random.uniform(
                -self.max_shift, self.max_shift - delays[0], 1)
            if self.verbose:
                print(f'Position: {(self.n_tpoints/2 + t0)*self.tstep}')
        gauss_mus = t0 + delays
        if np.abs(gauss_mus).max() > self.n_tpoints/2:
            raise ValueError(
                f'Time interval is too short for the event.\n'
                f'Increase the n_tpoints parameter')
        
        # this should be here, not in the loop below!
        if scint:
            scint_amp = self.scintillation(self.f_centers, self.ref_freq)
        else:
            scint_amp = np.ones(self.n_chan)
    
        for chan in range(self.n_chan):
            self.pulse[chan,:] =  self.pulse_profile(
                nt=self.n_tpoints,          # number of time samples
                width=width/self.tstep,     # gaussian width in samples
                f=self.f_centers[chan],     # frequency in MHz
                tau=tau/self.tstep,         # scatterring time at 1 GHz in samples
                t0=gauss_mus[chan],         # profile center shift in samples
                dm=dm,                      # dispersion measure in [pc cm-3]
                delta_freq=self.delta_freq, # channel width in MHz
                tsamp=self.tstep            # sampling time in seconds
            )
            
            # Spectral index
            pulse_chan = self.pulse[chan,:]
            pulse_chan = np.nan_to_num(pulse_chan)
            pulse_chan *= (self.f_centers[chan]/self.ref_freq)**sp_ind
            
            # Scintillation
            if scint:
                #scint_amp[chan] += np.random.uniform(1e-18, 0.05, 1).item()
                scint_amp[chan] += np.random.uniform(1e-18, 0.1, 1).item()
                pulse_chan *= scint_amp[chan]
            self.pulse[chan,:] = pulse_chan
            
        # Amplitude
        self.pulse = ampl * self.pulse / np.amax(self.pulse)

        # S/N
        sn = np.linalg.norm(scint_amp / np.amax(scint_amp)) * ampl

        # Noise
        if noise:
            self.pulse += np.random.randn(*self.pulse.shape)
        
        return self.pulse, sn
    
    def pulse_per_chan(
        self,
        t0=0.5,        # start in seconds
        ampl=[3,3,3,3],# amplitudes
        width=0.0005,  # width in seconds
        dm=50,         # dispersion measure in (pc cm−3)
        tau=0.004,     # scattering time at 1 GHz in seconds
        noise=False,   # add a Gaussian noise with sigma = 1 ampl
    ):
        """ Generate a pulse with certain amplitudes: nd.array [n_chan, n_tpoints]
        """
        t0 = t0/self.tstep - self.n_tpoints/2  # relative to center
        ampl = ampl[::-1]  # revert
        delays = self.get_delays(self.f_centers, dm) / self.tstep
                
        gauss_mus = t0 + delays
    
        for chan in range(self.n_chan):
            self.pulse[chan,:] =  self.pulse_profile(
                nt=self.n_tpoints,          # number of time samples
                width=width/self.tstep,     # gaussian width in samples
                f=self.f_centers[chan],     # frequency in MHz
                tau=tau/self.tstep,         # scatterring time at 1 GHz in samples
                t0=gauss_mus[chan],         # profile center shift in samples
                dm=dm,                      # dispersion measure in [pc cm-3]
                delta_freq=self.delta_freq, # channel width in MHz
                tsamp=self.tstep            # sampling time in seconds
            )
            
            pulse_chan = self.pulse[chan,:]
            pulse_chan = np.nan_to_num(pulse_chan)
            
            pulse_chan = ampl[chan] * pulse_chan / np.max(pulse_chan)
            self.pulse[chan,:] = pulse_chan
            

        # S/N
        sn = np.linalg.norm(ampl)

        # Noise
        if noise:
            self.pulse += np.random.randn(*self.pulse.shape)
        
        return self.pulse, sn
    
    def plot_pulse(self, pulse=None, title=None, n_xticks=16, n_yticks=4):
        """ Plotting the pulse 
            (use for a sufficient number of channels: 
             the size of the image corresponds to the number of channels)
        """
        if pulse is None:
            pulse = self.pulse
        
        plt.imshow(pulse[::-1,:])   # Higher frequencies at the top
        
        xr = np.linspace(0, pulse.shape[1], n_xticks)
        xl = [f'{x:.1f}' 
              for x in np.linspace(0, self.n_tpoints*self.tstep, n_xticks)]
        plt.gca().set_xticks(xr, labels=xl)
        yr = np.linspace(0, self.n_chan, n_yticks)
        yl = np.linspace(self.f_range[1], self.f_range[0], n_yticks)
        plt.gca().set_yticks(yr, labels=yl)
        plt.xlabel('Time, s')
        plt.ylabel('Frequency, MHz')
        plt.title(title)