#########1#########2#########3#########4#########5#########6#########7#########
"These are service functions, some are for the RATAN-600 data only"
import configparser, json
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import shift
from scipy.signal import butter, sosfiltfilt

def plot_chans(data, n=0, length=1, rec_length=4080, increment=0.000245):
    """Plot "length" records, each is a time series with "rec_length" elements
       and the "increment" time step; all channels in one figure.
       Typically can be used to print a "length" number of seconds starting
       from the "n"th second (if rec_length * increment ~ 1)
    """
    fig, ax = plt.subplots(figsize=(16, 5))
    start = round(n * rec_length)
    end =  round((n+length) * rec_length)
    record = data[:, start:end]
    x = np.array(range(record.shape[1])) * increment
    for chan in range(record.shape[0]):
        plt.plot(x, record[chan], label=f'chan {chan}')
    plt.xlabel('Time, s')
    plt.ylabel('Amplitude')
    plt.tight_layout()   
    return record


def plot_detail(
    data, start=None, end=None, vline=None, increment=0.000245, 
    sharey=True, figsize=None
):
    """Plot each channel in a separate subplot 
       (only for RATAN-600 4-channel records)"""
    if start is None:
        start = 0
        start_px = 0
    else:
        start_px = round(start / increment)
    if end is None:
        end = data.shape[1] * increment
        end_px = data.shape[1]
    else:
        end_px = round(end / increment)
    
    if figsize:
        fig, ax = plt.subplots(5, 1, figsize=figsize)
    else:
        fig, ax = plt.subplots(5, 1, figsize=(17, 8))
    record = data[:, start_px:end_px]
    x = np.array(range(record.shape[1]))*increment + start
    for chan in range(record.shape[0]):
        ax[0].plot(x, record[chan])
    
    bottom, top = ax[0].get_ylim()
    for i, axi in enumerate(ax[1:]):
        axi.plot(x, record[3-i])
        if sharey:
            axi.set_ylim(bottom=bottom, top=top)
        if vline:
            axi.axvline(vline, color='r', ls='--')
    
    ax[0].plot([], [], ' ', label='All channels')
    ax[1].plot([], [], ' ', label='4925 GHz')
    ax[2].plot([], [], ' ', label='4775 GHz')
    ax[3].plot([], [], ' ', label='4625 GHz')
    ax[4].plot([], [], ' ', label='4475 GHz')
    for axi in ax:
        axi.legend(loc='upper left')

    plt.xlabel('Time, s')
    fig.supylabel('Amplitude')
    plt.tight_layout()
    return fig, ax


def get_yrange(axes):
    "Get the range for shared y axes in subplots"
    bottom_out, top_out = np.inf, -np.inf
    for ax in axes:
        bottom, top = ax.get_ylim()
        if bottom < bottom_out: bottom_out = bottom
        if top > top_out: top_out = top
    return bottom_out, top_out


def lognormal_ampl(mu=3.5, sigma=1, lower_bound=2, upper_bound=np.inf):
    """Draws an amplitude from a lognormal distribution 
       within the bounds
    """
    ampl = -1
    while ampl<lower_bound or ampl>upper_bound:
       ampl = np.random.lognormal(mu, sigma)
    return ampl


def read_config():
   "A function to read the config file during training"
   config = configparser.ConfigParser()
   config.read('config.ini')
   tau_range = json.loads(config['Ranges']['tau'])
   dm_range = json.loads(config['Ranges']['dm'])
   sp_ind_range = json.loads(config['Ranges']['sp_ind'])
   ampl_mu = json.loads(config['Ranges']['ampl_mu'])
   ampl_sigma = json.loads(config['Ranges']['ampl_sigma'])
   ampl_bounds = json.loads(config['Ranges']['ampl_bounds'])
   lowres_range = json.loads(config['Ranges']['lowres_range'])
   highres_range = json.loads(config['Ranges']['highres_range'])
   sn_bound = json.loads(config['Ranges']['sn_bound'])
   return (
      tau_range, dm_range, sp_ind_range, ampl_mu, ampl_sigma, ampl_bounds,
      lowres_range, highres_range, sn_bound)


def dedisp_sum(data, dm, radiometer, noise_norm=True, fast=False):
    "Calculate a dedispersed total pulse"
    delays = (radiometer.get_delays(radiometer.f_centers, dm) / radiometer.tstep)
    d_data = data.copy()
    if fast:
        delays = np.int16(np.round(delays))
        for i in range(radiometer.n_chan):
            d_data[i] = np.roll(d_data[i], -delays[i])
    else:
        for i in range(radiometer.n_chan):
            d_data[i] = shift(data[i], -delays[i])
    if noise_norm:
        summed = np.sum(d_data, axis=0) / np.sqrt(radiometer.n_chan)
    else:
        #summed = np.sum(d_data, axis=0) / radiometer.n_chan
        summed = np.average(d_data, axis=0)
      
    return summed


def filter_noise(data, increment, bound, axis=1):
    "Low-pass Butterworth filter (8th order)"
    fs = 1 / increment  # sampling frequency
    sos = butter(N=4, Wn=bound, btype='lowpass', fs=fs, output='sos')
    data_filtered = np.empty_like(data)
    data_filtered = sosfiltfilt(sos, data, axis=axis)    
    return data_filtered


def plot_dmtime(
    data, radiometer, t_range=(0, 1), dm_range=(0, 1000), n_dms=None, fast=True,
    t_event=None, dm_event= None, title=None, n_xticks=5, n_yticks=5, figsize=(6.4, 6.4)
):
    """Plot DM-time diagram

    Args:
        data (np.array): Multichannel time series
        radiometer: an instance of the Radiometer class
        t_range (tuple, optional): time interval to be displayd, in seconds. Defaults to (0, 1).
        dm_range (tuple, optional): DM interval to be displayd. Defaults to (0, 1000).
        n_dms (int, optional): number of calculated DMs.
        fast (bool, optional): if fast, then descrete integer shift between DMs. Defaults to True. 
        t_event (float, optional): time of the event, if known. Defaults to None.
        dm_event (float, optional): DM of the event, if known. Defaults to None.
        title (str, optional): title of the figure. Defaults to None.
        n_xticks (int, optional): # xticks. Defaults to 5.
        n_yticks (int, optional): # yticks. Defaults to 5.
        figsize (tuple, optional): figure size in inches Defaults to (6.4, 6.4).

    Returns:
        fig, ax
    """
    if not n_dms:
        n_dms = (dm_range[1] -  dm_range[0]) // 10
    px_range = [round(x / radiometer.tstep) for x in t_range]
    if px_range[1] > data.shape[1]:
        px_range[1] = data.shape[1]
        t_range = np.array(t_range, dtype=np.float16)
        t_range[1] = data.shape[1] * radiometer.tstep
    npix_x = px_range[1] - px_range[0]
    reduce_factor = npix_x // n_dms
    npix_y = n_dms * reduce_factor
    
    dms = np.linspace(*dm_range, n_dms)
    dm_domain = np.empty(shape=(npix_y, npix_x))
    for i, dm in enumerate(dms):
        if fast:
            dds = dedisp_sum(data[:, px_range[0]:px_range[1]], dm, radiometer, fast=True)
        else:
            dds = dedisp_sum(data[:, px_range[0]:px_range[1]], dm, radiometer, fast=False)
        for j in  range(reduce_factor):
            dm_domain[i*reduce_factor+j, :] = dds
        
    dm_domain = dm_domain[::-1, :]
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(dm_domain)
    
    xr = np.linspace(0, npix_x, n_xticks)
    xl = [f'{x:.2f}'
           for x in np.linspace(*t_range, n_xticks)]
    plt.gca().set_xticks(xr, labels=xl)
    yr = np.linspace(0, npix_y,  n_yticks)
    yr = yr[::-1]
    yl = [f'{x:.0f}' 
          for x in np.linspace(*dm_range, n_yticks)]
    plt.gca().set_yticks(yr, labels=yl)
    plt.xlabel('Time, s')
    plt.ylabel(r'DM, $\rm pc\,cm^{-3}$')
    
    if t_event and dm_event:
        x_event = t_event/radiometer.tstep - px_range[0]
        y_event = npix_y - (dm_event-dm_range[0])/(dm_range[1]-dm_range[0])*npix_y
        ax.plot(x_event, y_event, 'r+', ms=20)
    if title:
        plt.title(title)
    
    return fig, ax