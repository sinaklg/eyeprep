import os 
import numpy as np
from scipy.sparse import diags, eye
from scipy.sparse.linalg import spsolve
from scipy.interpolate import interp1d


'''
written by myself, Mai 2024, use at your own peril.
Implement the Dai et al., 2021 CGTV algorithm
can be found [here](https://jov.arvojournals.org/article.aspx?articleid=2772700)
and matlab implementation [here](https://eeweb.engineering.nyu.edu/iselesni/eye-movement/)

Uses recommendations described in paper to estimate parameters alpha and beta:
they recommand using:
$\beta=4\sqrt{A}e^{5D}\sigma$ where A is average saccade amplitude and D average saccade duration, estimated from the data, $\sigma$ the average standard deviation of the fixation data, but should take into account sampling frequency: 
- if f <= 500Hz, use $\alpha=0.016f\sigma$ and $\beta=0.008f\sqrt{A}e^{5D}\sigma$
- if f > 500Hz, use $\alpha=(0.0032f+6.4)\sigma$ and $\beta=(0.0016f+3.2)\sqrt{A}e^{5D}\sigma$

To estimate average A, D and $sigma$ they recommend
- apply low pass differentiator with 10Hz freq cut-off
- classify saccades as ones above 10deg/s and more than 12ms
- combine 2 saccades if less than 20 samples between 2 consecutive ones

CGTV algo for Compound and Generalized Total Variation (using 1st and 3rd order derivatives).
'''


## RECOMMENDED PARAMS (see Paper)

'''
Three classes of params:
    - 1. parameter estimation alpha and beta for algorithm - prefix pe_
    - 2. saccade detection - prefix sd_
    - 3. post-processing/refinement - prefix pp_


Changes: 
    - Needed parameter for saccade peak vel different between species
    - Needed different parameter for how far around blink ignoring saccades (bc different algorithm for human and macaque blink detection)
'''

pe_freqCutoff = 10 #Hz used for inital velocity estimation with lowPass differentiator
pe_VT_sSac = 10 #deg/s velocity thresh for saccade start
pe_Tdur = 0.012 #s min sac dur for noise estimation
pe_Sisi = 20 #n_samples
pe_lam = 1 #smoothing tuner? not used

sd_VT_sSac = 30 #deg/s
sd_VT_eSac = 10 #deg/s

pp_Tisi = 0.040 #s minimum inter-saccadic interval (discards post-saccadic oscillations)
pp_Tdur = 12 #ms minimum saccade duration / change if wants microsaccades
pp_Sblink = 100 #n_samples minimum around blink to consider saccade #MD dur would make more sense
pp_maxSacDur = 0.080 #s

pp_peakVel = 800 #(macaque vs human: https://journals.physiology.org/doi/full/10.1152/jn.00312.2020)

## FUNCS

def calc_derivative_signal(sacc_signal, Fs):
    """
    This function calculates the derivative of the input saccade signal to 
    calculate the velocities of each sample.

    @param signal: the input saccade signal.
    @param Fs: the sampling rate (samples/second) of the input signal.
    
    @return vel: the derivative of the input signal, i.e. the velocities.
    """
    h = np.array([0.5, 0, -0.5])
    vel = np.convolve(sacc_signal, Fs*h, mode = 'same')
    vel[0] = 0
    vel[-1] = 0
    return vel

def mybwlabel(x):
    """
    Label connected ones in a binary array.

    Args:
    x (numpy array): Input binary array.

    Returns:
    numpy array: Labeled array.
    int: Number of connected ones.
    """
    y = np.cumsum(np.insert(np.abs(np.diff(x)), 0, 1))
    if x[0]:
        y[y % 2 == 0] = 0
        y[y % 2 == 1] = (y[y % 2 == 1] + 1) // 2
    else:
        y[y % 2 == 1] = 0
        y[y % 2 == 0] = y[y % 2 == 0] // 2
    n = np.max(y)
    return y, n

def calculate_moving_average_window_size(sampling_rate, cutoff_frequency):
    window_duration = 0.443 / cutoff_frequency
    window_size = int(window_duration * sampling_rate)
    return window_size

def moving_average_filter(signal, window_size, mode='same'):
    """
    Applies a moving average filter to a signal.

    Args:
    signal (numpy array): Input signal.
    window_size (int): The number of samples over which to average.

    Returns:
    numpy array: The filtered signal.
    """
    kernel = np.ones(window_size) / window_size
    filtered_signal = np.convolve(signal, kernel, mode=mode)
    return filtered_signal

def get_alpha(Fs, sigma):
    """
    Compute alpha parameter of CGTV algo

    Args:
    Fs: sampling frequency in Hz
    sigma: standard deviation of fixation position
    ampl: average saccade amplitude in deg
    dur: average saccade duration in secs

    Returns:
    float: alpha
    """
    if Fs <= 500:
        return 0.016*Fs*sigma
    else:
        return (0.0032*Fs+6.4)*sigma
    
def get_beta(Fs, sigma, ampl, dur):
    """
    Compute beta parameter of CGTV algo

    Args:
    Fs: sampling frequency in Hz
    sigma: standard deviation of fixation position
    ampl: average saccade amplitude in deg
    dur: average saccade duration in secs

    Returns:
    numpy array: beta
    """
    if Fs <=500:
        return 0.008*Fs*np.sqrt(ampl)*np.exp(5*dur)*sigma
    else:
        return (0.0016*Fs+3.2)*np.sqrt(ampl)*np.exp(5*dur)*sigma

def interpolate_nans(y):
    """
    Interpolate NaN values in an array.

    Args:
    y (numpy array): Input array with NaNs.

    Returns:
    numpy array: Array with NaNs interpolated.
    """
    if np.isnan(y).any():
        nans = np.isnan(y)
        x = lambda z: z.nonzero()[0]
        y[nans] = interp1d(x(~nans), y[~nans], kind='linear', fill_value='extrapolate')(x(nans))
        return y, nans
    else:
        return y, np.array([], dtype=int)

def cgtv(y, alpha, beta, Nit):
    """
    Compound and Generalized Total Variation (CGTV) algorithm, handling NaNs.

    Args:
    y (numpy array): Input data.
    alpha (float): Regularizer parameter (for 1st order derivative component).
    beta (float): Regularizer parameter (for 3rd order derivative component).
    Nit (int): Number of iterations.

    Returns:
    numpy array: Output data.
    """

    EPS = 1E-10  # Smoothed penalty function
    psi = lambda x: np.sqrt(x**2 + EPS)

    y, nans = interpolate_nans(y.flatten())
    N = len(y)

    e = np.ones(N)
    D1 = diags([-e, e], [0, 1], shape=(N-1, N))
    D3 = diags([-e, 3*e, -3*e, e], [0, 1, 2, 3], shape=(N-3, N))
    I = eye(N)

    x = y.copy()  # Initialization

    for _ in range(Nit):
        Lam1 = diags([alpha / psi(np.diff(x, n=1))], [0], shape=(N-1, N-1))
        Lam3 = diags([beta / psi(np.diff(x, n=3))], [0], shape=(N-3, N-3))
        temp = I + D1.T @ Lam1 @ D1 + D3.T @ Lam3 @ D3
        x = spsolve(temp, y)

    x[nans] = np.nan
    return x

def calculateVelocity(signal, Fs):
    """
    This function calculates the derivative of the input saccade signal to 
    calculate the velocities of each sample.

    @param signal: the input saccade signal.
    @param Fs: the sampling rate (samples/second) of the input signal.
    
    @return vel: the derivative of the input signal, i.e. the velocities.
    """
    h = np.array([0.5, 0, -0.5])
    vel = np.convolve(signal, Fs*h, mode = 'same')
    vel[0] = np.nan
    vel[-1] = np.nan
    return vel

def calculateAcceleration(signal, Fs):
    """
    This function calculates the derivative of the input saccade signal to 
    calculate the velocities of each sample.

    @param signal: the input saccade signal.
    @param Fs: the sampling rate (samples/second) of the input signal.
    
    @return vel: the derivative of the input signal, i.e. the velocities.
    """
    fvel = np.array([0.5, 0, -0.5])*Fs
    facc = np.convolve(fvel, fvel)
    acc = np.convolve(signal, facc, mode = 'same')
    acc[0] = np.nan
    acc[-1] = np.nan
    return acc


class cgtvParser(object):

    '''
    Assumes continuous input, and np.nan when in blink / no reliable data
    pos_x, pos_y in deg
    '''

    def __init__(self, 
                 pos_x, pos_y, sampling_frequency=1000, 
                 pe_freqCutoff=pe_freqCutoff, pe_VT_sSac=pe_VT_sSac,
                 pe_Tdur=pe_Tdur, pe_Sisi=pe_Sisi, pe_lam=pe_lam,
                 sd_VT_sSac=sd_VT_sSac, sd_VT_eSac=sd_VT_eSac,
                 pp_Tisi=pp_Tisi, pp_Tdur=pp_Tdur, pp_Sblink=pp_Sblink,
                 pp_peakVel=pp_peakVel, pp_maxSacDur=pp_maxSacDur,
                 debug=False):
        self.Fs = sampling_frequency #Hz
        self.freqCutoff = pe_freqCutoff 
        self.VTnoise = pe_VT_sSac
        self.pe_Tdur = pe_Tdur
        self.Sisi = pe_Sisi #n_samples
        self.lam = pe_lam
        self.VT_sSac = [sd_VT_sSac]
        self.VT_eSac = sd_VT_eSac
        self.Tisi = pp_Tisi
        self.pp_Tdur = pp_Tdur
        self.Sblink = pp_Sblink #n_samples after blinks
        self.peakVel = pp_peakVel
        self.maxSacDur = pp_maxSacDur #s
        self.debug = debug

        self.dataOrig = np.stack((pos_x, pos_y, np.array(range(len(pos_x)))/self.Fs), axis=0) #shape: 3*n, third_dim is time of sample in secs
        self.step = None
        self.N = self.dataOrig.shape[-1]
        self.dataProc = np.empty((7, self.N))
        self.dataProc[2] = self.dataOrig[2]


    def step1_getDenoisedPositions(self):

        self.step = 'step 1 of 2'
        for i in range(2): # separately for x and y
            if self.debug:
                print(self.step, 'processing %d'%i)
            xx = self.dataOrig [i].copy()
            fdx = calc_derivative_signal(xx, self.Fs) #filtered velocity
            fdx = moving_average_filter(fdx, calculate_moving_average_window_size(self.Fs, self.freqCutoff))
            tmp1 = np.abs(fdx) >= self.VTnoise #threshold - is_saccade
            aa, bb = mybwlabel(tmp1) 

            for b in range(bb+1):
                bix = np.where(b == aa)[0]
                if len(bix) < (self.pe_Tdur*self.Fs):
                    tmp1[bix] = 0
            # print(tmp1.sum())

            idx = tmp1[1:].astype(np.int32) - tmp1[:-1].astype(np.int32) 
            SP = np.where(idx == 1)[0] #StartPoint #MD changed
            if tmp1[0] == 1:
                SP = np.insert(SP, 0, 0)

            EP = np.where(idx == -1)[0] + 1 #EndPoint
            if tmp1[-1] == 1: # bug in their code
                EP = np.append(EP, self.N - 1)

            for EPi in range(len(EP) - 1):
                if (SP[EPi + 1] - EP[EPi]) < self.Sisi:
                    tmp1[EP[EPi]:SP[EPi + 1]] = 1

            if self.debug:
                print(self.step, len(SP), 'saccades detected first pass')

            # estimate noise, amplitude and duration
            fixation, nfix = mybwlabel(tmp1==0)
            xfix = xx.copy()
            for f in range(nfix+1):
                fix_ix = np.where(fixation == f)[0]
                xfix[fix_ix] = xfix[fix_ix] - np.nanmean(xfix[fix_ix])
            sigma = np.nanstd(xfix[~tmp1]) #MD higher sigma for x when clearly should be y, but prb more saccades along x.

            tmp3 = np.abs(fdx)
            tmp3[~tmp1] = 0
            tmp4, nsacc = mybwlabel(tmp3!=0)

            if self.debug:
                print(self.step, nsacc, 'saccades detected after pooling') #MD why different number of saccades? Bc pooled if isi under 20ms

            if nsacc == 0:
                amp_avg = 1
                dur_avg = 0
            else:
                amp_cum = np.sum(tmp3)/self.Fs
                amp_avg = amp_cum/nsacc
                dur = np.zeros(nsacc)
                for nsaci in range(1, nsacc+1): #index careful discard 0
                    dur[nsaci-1] = np.sum((tmp4 == nsaci))
                dur_avg = np.mean(dur)/self.Fs #in s
            alpha = self.lam*get_alpha(self.Fs, sigma)
            beta = self.lam*get_beta(self.Fs, sigma, amp_avg, dur_avg)
            
            if self.debug:
                print(self.step, 'sigma: %3.2f ampl %3.2f, dur %3.2f'%(sigma, amp_avg, dur_avg))

            denoised_xx = cgtv(xx, alpha, beta, Nit=20)
            self.dataProc[i] = denoised_xx


    def step2_extractSaccades(self):

        self.step = 'step 2 of 2'
        if len(self.VT_sSac) == 1:
            self.VT_sSac = np.ones(2) * self.VT_sSac # same for x and y
        pos_x, pos_y = self.dataProc[0, :], self.dataProc[1, :]
        time_s = self.dataProc[2, :]
        time_ms = time_s*1000
        vel_x = calculateVelocity(pos_x, self.Fs)
        vel_y = calculateVelocity(pos_y, self.Fs)
        acc_x = calculateAcceleration(pos_x, self.Fs)
        acc_y = calculateAcceleration(pos_y, self.Fs)
        vel = np.sqrt(vel_x**2+vel_y**2)
        acc = calculateVelocity(vel, self.Fs)
        self.dataProc[3, :] = vel_x
        self.dataProc[4, :] = vel_y
        self.dataProc[5, :] = acc_x
        self.dataProc[6, :] = acc_y
        outside_idx = (vel_x/self.VT_sSac[0])**2 +(vel_y/self.VT_sSac[1])**2
        outside_idx = (outside_idx>=1)
        outside_idx = outside_idx.astype(np.int32)
        start_idx = np.where(np.diff(np.insert(outside_idx, 0, 0)) == 1)[0] #MD adding the -1 seems better corresponds to data?
        end_idx = np.where(np.diff(np.append(outside_idx, 0)) == -1)[0]
        nsacs = len(end_idx)

        tmpSaccades = [dict()]*nsacs
        saccfilter = np.ones(nsacs)

        durs = []
        for s in range(nsacs):

            # included in previous sac
            if s != 0 and s:
                if start_idx[s] <= np.max(end_idx[0:s]):
                    saccfilter[s] = 0

            # follows a blink within Tblink samples
            if s != 0:
                if np.sum(np.isnan(vel[np.max((0, start_idx[s]-self.Sblink)):start_idx[s]])) > 0:
                    saccfilter[s] = 0 #MD could identify ending (thresh under Tvel_end and add to blink)

            # velocity threshold end of saccade
            lower_bnd = end_idx[s]
            upper_bnd = int(np.min((lower_bnd+self.maxSacDur*self.Fs, self.N)))
            first_spl_esacc = np.where(vel[lower_bnd:upper_bnd] < self.VT_eSac)[0]
            if (len(first_spl_esacc) == 0) or (np.sum(np.isnan(vel[lower_bnd:upper_bnd])) > 0):
                end_idx[s] = end_idx[s] + len(np.arange(lower_bnd, upper_bnd)) - 1
            else:
                end_idx[s] = end_idx[s] + first_spl_esacc[0] - 1
            
            # this saccade followed by a blink within Tblink samples
            if s != 0:
                if np.sum(np.isnan(vel[end_idx[s]:np.min((end_idx[s]+self.Sblink, self.N))])) > 0:
                    saccfilter[s] = 0 #MD should also consider as blink
                    
            # this saccade is actually a blink
            if np.sum(np.isnan(vel[start_idx[s]: end_idx[s]])) > 0:
                saccfilter[s] = 0 #MD should consider blink
                
            # this saccade satisfies minimum duration
            dur = time_ms[end_idx[s]]-time_ms[start_idx[s]]
            durs.append(dur)
            if dur < self.pp_Tdur:
                saccfilter[s] = 0
            
            # this saccade satisfies minimal intersaccadic interval
            if s != 0:
                k = s-1
                while saccfilter[k] == 0 and k > 1:
                    k = k-1
                if saccfilter[k] == 0 and k == 1:
                    isi = np.nan
                else:
                    isi = time_ms[start_idx[s]] - time_ms[end_idx[k]]
                    if isi < (self.Tisi*1000):
                        saccfilter[s] = 0
            else:
                isi = np.nan
            ampl = np.sqrt((pos_x[end_idx[s]]-pos_x[start_idx[s]])**2+(pos_y[end_idx[s]]-pos_y[start_idx[s]])**2)
            try: 
                pvel = np.abs(np.max(vel[start_idx[s]:end_idx[s]]))
            except:
                pvel = 0
            angle = np.arctan2(pos_y[end_idx[s]]-pos_y[start_idx[s]], pos_x[end_idx[s]]-pos_x[start_idx[s]])
            if pvel > self.peakVel:
                saccfilter[s] = 0
            tmpSaccades[s] = {'dur': dur, 'ampl': ampl, 'angle': angle, 'peak_vel':  pvel, 'isi': isi,
                           'tstart': time_s[start_idx[s]], 'tend': time_s[end_idx[s]],
                           'ixstart': start_idx[s], 'ixend': end_idx[s],
                           'start_x': pos_x[start_idx[s]], 'start_y': pos_y[start_idx[s]], 
                           'end_x': pos_x[end_idx[s]], 'end_y':pos_y[end_idx[s]]} #using denoised position for sac estimates


        self.dSaccades = [] # final dictionary of saccades
        ixok = np.where(saccfilter)[0]
        for i in ixok:
            self.dSaccades.append(tmpSaccades[i])

        self.mov_type = np.zeros(self.N) #array to store in which case one is
        for s in self.dSaccades:
            self.mov_type[s['ixstart']: s['ixend']] = 1

        print('Total number of saccades detected: %d'%len(self.dSaccades))



    def parse(self):

        '''
        outputs:
        - dictionary of saccades with info
        - an array to represent if in saccade (1) or not
        - the denoised eye positions (x, y, and time)
        '''

        self.step1_getDenoisedPositions()
        self.step2_extractSaccades()

        return self.dSaccades, self.mov_type, self.dataProc
