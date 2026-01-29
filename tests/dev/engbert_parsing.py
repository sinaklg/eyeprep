import os 
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

'''
created my marianne.duyck@gmail.com Aug 2022
after Engbert & Kliegl, 2003, Vision Research and Engbert & Mergenthaler, 2006, PNAS (algorithm to detect microsaccades).
last edited Aug 2022
'''


colors = {'saccades': 'royalblue', 'microsaccades': 'lightskyblue', 'blinks': 'dimgrey', 'fixations': 'silver',
'x': 'palegreen', 'y': 'slateblue', 'fx': 'g', 'fy': 'b'}


def getAngle(start_x, start_y, end_x, end_y):
    return np.arctan2(end_y-start_y, end_x-start_x)

def wrapTo2pi(rad_values):
#rad_values is a np array in rad
    if type(rad_values)!= np.ndarray: rad_values = np.asarray(rad_values)
    positiveInput = (rad_values > 0)
    rad_values = rad_values%(2*np.pi)
    rad_values[(rad_values == 0) & positiveInput] = 2*np.pi
    return rad_values

def get_rate(nb_events, nb_samples, samplingRate):
    return nb_events/nb_samples*samplingRate

def ax_polar_plot(fig, dat_rad, ticks_step=5, bin_size_deg=12, subplot=111, color=[0.5]*3):
    degrees = np.rad2deg(wrapTo2pi(dat_rad))
    #a, b = np.histogram(degrees, bins=np.arange(-bin_size_deg/2, 360+bin_size_deg/2, bin_size_deg))
    a, b = np.histogram(degrees, bins=int(360/bin_size_deg))
    centers = np.deg2rad(np.ediff1d(b)//2 + b[:-1])
    max_hist = np.max(a)
    ax = fig.add_subplot(subplot, projection='polar')
    ax.grid(linewidth=0.3, linestyle='--', zorder=0.5)
    ax.set_axisbelow(True)
    ax.bar(centers, a, width=np.deg2rad(bin_size_deg), bottom=0.0, color=color, edgecolor='None')
    min_r, max_r = 0, max_hist+0.15*max_hist
    #ax.set_rticks(np.arange(0, np.around(max_r, -1), ticks_step))
    ax.set_rticks(np.linspace(0, np.around(max_r, -1), ticks_step))
    ax.set_rlim(min_r, max_r)
    ax.set_rlabel_position(45)
    return ax

def ax_hist(ax, data, color_hist, color_median='r', alpha=1, fmt='%d'):
    ax.hist(data, color=color_hist, alpha=alpha)
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.axvline(x=np.median(data), color=color_median, ls='--')
    if not(np.isnan(np.nanmedian(data))):
        print(np.nanmedian(data))
        ax.annotate('median = '+fmt%(np.nanmedian(data)), xy= (ax.get_xlim()[1]*1/2, ax.get_ylim()[1]*1/2), color=color_median)
    return ax

def plot_summary(fig, events, data_len, samplingRate):
    ampl_sac = extractFromEvents(events, 'ampl', 'saccades')
    dur_fx = extractFromEvents(events, 'dur', 'fixations')
    dur_bk = extractFromEvents(events, 'dur', 'blinks')
    angle_sac = extractFromEvents(events, 'angle', 'saccades')
    peak_vel_sac =  extractFromEvents(events, 'peak_vel', 'saccades')
    if 'microsaccades' in events.keys():
        angle_usac = extractFromEvents(events, 'angle', 'microsaccades')
        peak_vel_usac =  extractFromEvents(events, 'peak_vel', 'microsaccades')
        ampl_usac = extractFromEvents(events, 'ampl', 'microsaccades')


    ax0 = ax_polar_plot(fig, angle_sac, bin_size_deg=12, subplot=231, color=colors["saccades"])
    ax0.set_title('saccades\n rate: %3.2f [Hz]'%(get_rate(len(events["saccades"]), data_len, samplingRate)))

    if 'microsaccades' in events.keys():
        ax1 = ax_polar_plot(fig, angle_usac, bin_size_deg=12, subplot=232, color=colors['microsaccades'])
        ax1.set_title('microsaccades\n rate: %3.2f [Hz]'%(get_rate(len(events["microsaccades"]), data_len, samplingRate)))

    ax2 = fig.add_subplot(233)
    ax2.scatter(ampl_sac, peak_vel_sac, s=2, color=colors["saccades"])
    if 'microsaccades' in events.keys():
        ax2.scatter(ampl_usac, peak_vel_usac, s=2, color=colors["microsaccades"])
    ax2.set_ylim([1, 1000])
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_ylim([50, 1500])
    ax2.set_xlim([0.2, 50])
    ax2.set_xlabel('amplitude [dva]')
    ax2.set_ylabel('peak velocity [dva/s]')
    ax2.spines.right.set_visible(False)
    ax2.spines.top.set_visible(False)

    ax3 = ax_hist(fig.add_subplot(234), dur_bk, colors["blinks"])
    ax3.set_xlabel('duration [ms]')
    ax3.set_ylabel('count')
    ax3.set_title('blinks\n rate [Hz]: %3.2f'%(get_rate(len(events["blinks"]), data_len, samplingRate)))

    ax4 = ax_hist(fig.add_subplot(235), dur_fx, colors["fixations"])
    ax4.set_xlabel('duration [ms]')
    ax4.set_ylabel('count')
    ax4.set_title('fixations \n rate [Hz]: %3.2f'%(get_rate(len(events["fixations"]), data_len, samplingRate)))

    ax5 = ax_hist(fig.add_subplot(236), ampl_sac, colors["saccades"], fmt='%3.2f')
    ax5.set_xlabel('amplitude [dva]')

    return fig

def plot_event(events, event_type, event_number, velocities_x, velocities_y, denoised_positions_x, denoised_positions_y, nsamples_around=200, velocities_x_filtered=[], velocities_y_filtered=[], other_position_x=[], other_position_y=[]):
    '''
    event_type: blinks, fixations, microsaccades, saccades
    velocities and positions are for whole experiment, not only this event
    do not plot events that could be present in the vicinity
    '''
    f, ax = plt.subplots(2, 2, figsize=(8, 8))
    delta = nsamples_around
    if 'ixstart' in events[event_type][event_number].keys():
        which_start = 'ixstart'
        which_end = 'ixend'
    else:
        which_start = 'tstart'
        which_end = 'tend'
    start, end = events[event_type][event_number][which_start], events[event_type][event_number][which_end] 
    ax[0, 0].plot(range(start-delta, end+delta), denoised_positions_x[start-delta:end+delta], color=colors['x'])
    ax[0, 0].plot(range(start-delta, end+delta), denoised_positions_y[start-delta:end+delta], color=colors['y'])
    if len(other_position_x) != 0:
        ax[0, 0].plot(range(start-delta, end+delta), other_position_x[start-delta:end+delta], color=colors['fx'])
    if len(other_position_y) != 0:
        ax[0, 0].plot(range(start-delta, end+delta), other_position_y[start-delta:end+delta], color=colors['fy'])
    ax[0, 0].axvspan(start, end, color=colors[event_type], alpha=0.2)
    ax[0, 1].plot(range(start-delta, end+delta), velocities_x[start-delta:end+delta], color=colors['x'])
    ax[0, 1].plot(range(start-delta, end+delta), velocities_y[start-delta:end+delta], color=colors['y'])
    if len(velocities_x_filtered) != 0:
        ax[0, 1].plot(range(start-delta, end+delta), velocities_x_filtered[start-delta:end+delta], color='r', lw=1, ls='--')
    if len(velocities_y_filtered) != 0:
        ax[0, 1].plot(range(start-delta, end+delta), velocities_y_filtered[start-delta:end+delta], color='r', lw=1, ls='--')

    ax[0, 1].axvspan(start, end, color=colors[event_type], alpha=0.2)
    ax[1, 0].plot(denoised_positions_x[start-delta:end+delta], denoised_positions_y[start-delta:end+delta], 'k', lw=0.5)
    ax[1, 0].plot(denoised_positions_x[start:end], denoised_positions_y[start:end], 'r', lw=1)
    ax[1, 1].plot(velocities_x[start-delta:end+delta], velocities_y[start-delta:end+delta], 'k', lw=0.5)
    ax[1, 1].plot(velocities_x[start:end], velocities_y[start:end], 'r', lw=1)

    ax[1, 1].axis('equal')
    ax[0, 0].set_title('position', fontsize=12)
    ax[0, 1].set_title('velocity', fontsize=12)
    ax[0, 0].set_ylabel('1D', fontsize=12)
    ax[1, 0].set_ylabel('2D', fontsize=12)
    f.suptitle('%s %d'%(event_type, event_number))

    return f

def plot_events_in_timewindow(ax, eventsInWin, start, end, denoised_positions_x, denoised_positions_y):
    ax.plot(range(start, end), denoised_positions_x[start:end], color=colors['x'])
    ax.plot(range(start, end), denoised_positions_y[start:end], color=colors['y'])
    for ev_type in eventsInWin.keys():
        if eventsInWin[ev_type] != []:
            for ev in eventsInWin[ev_type]:
                if 'ixstart' in ev.keys():
                    ax.axvspan(ev['ixstart'], ev['ixend'], color=colors[ev_type], alpha=0.3)
                else:
                    ax.axvspan(ev['tstart'], ev['tend'], color=colors[ev_type], alpha=0.3)

    ax.set_ylim((-15, 15))
    return ax

def extractFromEvents(eventsDict, variable, eventType):
    '''
    eg extractFromEvents(events, 'ampl', 'saccades') will return a list of amplitudes of all saccades
    '''
    return [a[variable] for a in eventsDict[eventType]]


def plot_all_events(eventsDict, event_type, savedir, nsamples_around=50):
    n=0
    for saccade in events[event_type]:
        f = plot_event(events, event_type, n, vel_x, vel_y, denoised_x, denoised_y, nsamples_around=nsamples_around)
        f.savefig(os.path.join(savedir, '%s_%03d.pdf'%(event_type, n)))
        n+=1

def extractEventsInTimeWindow(eventsDict, start, end):
    evts = {'blinks':[], 'saccades':[], 'microsaccades':[]}

    for ev_type in eventsDict.keys():
        for ev in eventsDict[ev_type]:
            if 'ixstart' in ev.keys():
                which_start = 'ixstart'
                which_end = 'ixend'
            else:
                which_start = 'tstart'
                which_end = 'tend'
            if ev_type != 'fixations':
                if ev[which_start] > start and ev[which_end]<end:
                    evts[ev_type].append(ev)
                elif ev[which_end] > start and ev[which_end] < end and ev[which_start] < start:
                    evts[ev_type].append(ev)
                    evts[ev_type][-1][which_start] = start
                elif ev[which_end] > end and ev[which_start] > start and ev[which_start] < end:
                    evts[ev_type].append(ev)
                    evts[ev_type][-1][which_end]=end
                else:
                    pass
    return evts


def extract_fixation_positions(eventsDict, pos_x, pos_y, limits=[None, None, None, None]):
    '''
    limits are left, right, bottom, top in same units as eye coordinates
    '''
    fixations = np.ones((len(pos_x), 2))*np.nan
    for f in eventsDict['fixations']:
        start = f['tstart']
        end = f['tend']
        fixations[start:end, 0] = pos_x[start:end]
        fixations[start:end, 1] = pos_y[start:end]
    if not None in limits:
        fixations[np.logical_and(fixations[:, 0] <= limits[0], fixations[:, 0] >= limits[1]), 0] = np.nan
        fixations[np.logical_and(fixations[:, 1] <= limits[2], fixations[:, 1] >= limits[3]), 1] = np.nan

    return fixations


def plot_position_heatmap(ax, pos_x, pos_y, min_x, max_x, min_y, max_y, nb_bins_x, cmap=plt.cm.viridis):
    nb_bins_y = int(round(nb_bins_x*(max_y-min_y)/(max_x-min_x)))
    ax.hist2d(pos_x, pos_y, bins=[np.linspace(min_x,max_x,n_bins_x),np.linspace(min_y,max_y)], density=True, cmap=plt.cm.viridis)
    return ax

class EngbertParser(object):

    def __init__(self, samplingRate=1000, minSacDur=6, threshBlink=35, velFactor=5, amplMaxMicroSac=0, verbose=True):
        '''
        in args:
            - samplingRate of data in Hz
            - minSacDur is in ms, 6 ms is default of their algorithm
            - threshBlink is in dva (could be better to use pupil to detect blinks), set by me
            - velFactor is their lambda, multiplicative factor of the noise in the data that delimits ellipse around fixation, their default is 5

        improvements:
            - also that was mainly designed to detect microsaccades in streams of fixation, 
            and clearly not optimal to detect all things (add nb of consecutive samples below thresh to stop event,
            compute noise level only on periods that are fixation cf big velocity artifacts around blinks, most currenly detected microsaccades are around blinks).
            at a first pass some events are slighly misestimated and keep in mind that blinks includes missing samples, monkey looking super far etc.
            !!! Should make sure to get a good send of efficacy by looking at events cf plot_event func.
            - prb best approach would be 1) reliably extract blinks (including eye closing and reopening phase) 2) extract saccades (lower velocity threshold, bc some saccades are currently ignored/shortened) 3) compute moving noise from fixation segments and extract microsaccades
            - using pupil to detect blinks
            - array like operations i.of separate for x and y
            - have denoised_positions correct positions (now offsets bc boudary effects, don't use as is)


        notes: 
            - couldn't get good results for both saccades and microsaccades with their defaults (saccades shortened and microsaccades not detected),
            so added a savgol filter on velocities, with set params, to help make sort of work this single algorithm to detect both micro and not micro saccades
            - fixations returned are fixations between saccades and microsaccades, to get standard fixations - that include microsaccades, have to use get_long_fixations.
        not well made ==> use showMeTheWay
        '''
        self.samplingRate = samplingRate
        self.sampleDur_ms = 1/samplingRate*1000
        self.minNbSamples = int(minSacDur/self.sampleDur_ms)
        if not(hasattr(threshBlink, '__len__')):
            self.threshBlink = [threshBlink, threshBlink]
        else:
            self.threshBlink = threshBlink
        self.velFactor = velFactor
        self.uSacMaxAmpl = amplMaxMicroSac
        self.verbose = verbose

    def get_smoothedVelocity(self, positions):
        kernel=[1, 1, 0, -1, -1]
        # easy way to get moving averaged estimated velocity is to use convolution
        # returns array of length len(positions)
        return self.samplingRate/6 * np.convolve(positions, kernel, 'valid') # MD careful returns array of length: len(input)-len(filter)+1

    def get_denoisedPositions(self, starting_position, velocities):
        return starting_position+1/self.samplingRate*np.cumsum(velocities)

    def get_noise(self, velocities):
        return self.velFactor * np.sqrt(np.median(velocities**2)-np.median(velocities)**2)

    def in_saccade(self, velocities_x, velocities_y, radius_threshold_x, radius_threshold_y):
        '''
        radius_thresholds are estimated by get_noise.
        '''
        return ((velocities_x/radius_threshold_x)**2 + (velocities_y/radius_threshold_y)**2) > 1

    def presegment_events(self, in_saccade, denoised_positions_x, denoised_positions_y):
        idx =  np.where(np.logical_or(in_saccade, np.logical_or(np.abs(denoised_positions_x)>self.threshBlink[0], np.abs(denoised_positions_y)>self.threshBlink[1])))[0] #MD added difference x and y
        ix = 0
        dur = 1
        a = 0
        tmp_evts = []
        nevts = 0
        while ix < len(idx)-1:
            if idx[ix+1]-idx[ix]==1:
                dur+=1
            else:
                if dur>=self.minNbSamples:
                    nevts += 1
                    b = ix
                    # tmp_evts.append([idx[a]-1, idx[b]] if a != 0 else [idx[a], idx[b]]) #MD correct
                    tmp_evts.append([idx[a]-1, idx[b]] if a != 0 else [idx[a], idx[b]])
                a = ix + 1
                dur = 1
            ix += 1
        return tmp_evts

    def segment_blinks_only(self, positions_x, positions_y):

        vel_x, vel_y = self.get_smoothedVelocity(positions_x), self.get_smoothedVelocity(positions_y)
        denoised_positions_x = self.get_denoisedPositions(positions_x[0], vel_x)
        denoised_positions_y = self.get_denoisedPositions(positions_y[0], vel_y)
        noise_x, noise_y = self.get_noise(vel_x), self.get_noise(vel_y)
        if self.verbose: print('noise x: ', noise_x, ' noise y: ', noise_y)
        in_saccade = self.in_saccade(vel_x, vel_y, noise_x, noise_y)
        pre_events = self.presegment_events(in_saccade, denoised_positions_x, denoised_positions_y)
        events = {x: [] for x in ["blinks"]}

        for event in pre_events:
            tstart, tend = event[0], event[1]
            samples = np.empty((tend-tstart, 2))
            samples[:, 0] = denoised_positions_x[tstart:tend]
            samples[:, 1] = denoised_positions_y[tstart:tend]
            dur = tend-tstart+1
            start_x = denoised_positions_x[tstart]
            start_y = denoised_positions_y[tstart]
            end_x = denoised_positions_x[tend]
            end_y = denoised_positions_y[tend]
            angle = np.arctan2(end_y-start_y, end_x-start_x)
            # if np.sum(np.max([np.abs(denoised_positions_x[tstart:tend]), np.abs(denoised_positions_y[tstart:tend])])>self.threshBlink) >= 1:
            if np.sum([np.max(np.abs(denoised_positions_x[tstart:tend]))>self.threshBlink[0], \
                       np.max(np.abs(denoised_positions_y[tstart:tend]))>self.threshBlink[1]])>= 1:

                events['blinks'].append({'ixstart': tstart,
                                        'ixend': tend,
                                        'tstart': tstart/self.samplingRate,  # stored time should be dpx one
                                        'tend': tend/self.samplingRate,
                                        'dur': dur})
        return events['blinks']

    def segment_most(self, presegmented_events, denoised_positions_x, denoised_positions_y, velocities_x, velocities_y):
        if self.uSacMaxAmpl > 0:
            events = {x: [] for x in ["saccades", "fixations", "blinks", "microsaccades"]}
        else:
            events = {x: [] for x in ["saccades", "fixations", "blinks"]}

        for event in presegmented_events:
            tstart, tend = event[0], event[1]
            samples = np.empty((tend-tstart, 2))
            samples[:, 0] = denoised_positions_x[tstart:tend]
            samples[:, 1] = denoised_positions_y[tstart:tend]
            dur = tend-tstart+1
            start_x = denoised_positions_x[tstart]
            start_y = denoised_positions_y[tstart]
            end_x = denoised_positions_x[tend]
            end_y = denoised_positions_y[tend]
            angle = np.arctan2(end_y-start_y, end_x-start_x)
            if np.sum(np.max([np.abs(denoised_positions_x[tstart:tend]), np.abs(denoised_positions_y[tstart:tend])])>self.threshBlink) >= 1:
                events['blinks'].append({'tstart': tstart,  # stored time should be dpx one
                                        'tend': tend,
                                        'dur': dur})
            else:
                ampl = np.sqrt((end_y-start_y)**2+(end_x-start_x)**2)
                if ampl > 0:
                    this_sac = {'tstart': tstart,
                                'tend': tend,
                                'dur': dur,
                                'ampl':ampl,
                                'start_x': start_x,
                                'start_y': start_y,
                                'end_x': end_x,
                                'end_y': end_y,
                                'peak_vel': np.max([np.abs(velocities_x[tstart:tend]), np.abs(velocities_y[tstart:tend])]),
                                'angle': angle}
                    if ampl > self.uSacMaxAmpl:
                        events['saccades'].append(this_sac)
                    else:
                        print(ampl, this_sac)
                        events['microsaccades'].append(this_sac)
        if self.verbose:
            print('\nblinks %d\nsaccades %d\nmicrosaccades %d\n'%(len(events['blinks']), len(events['saccades']), len(events['microsaccades']) if self.uSacMaxAmpl else 0))
        return events


    def segment_fixations(self, presegmented_events, other_events, data_len, denoised_positions_x, denoised_positions_y):
        fx_evts = []
        if presegmented_events[0][0]>2: # fix before first event
            fx_evts.append([0, presegmented_events[0][0]])

        for iev in range(len(presegmented_events)-1):
            fx_evts.append([presegmented_events[iev][1], presegmented_events[iev+1][0]])

        if presegmented_events[-1][1]<(data_len-2):
            fx_evts.append([presegmented_events[-1][1], data_len])

        for f in fx_evts:
            tstart = f[0]
            tend = f[1]
            dur = tend-tstart+1
            samples = np.empty((tend-tstart, 2))
            samples[:, 0] = denoised_positions_x[tstart:tend]
            samples[:, 1] = denoised_positions_y[tstart:tend]
            this_fix = {'tstart': tstart,
                        'tend': tend,
                        'dur': dur,
                        'x': np.median(samples[:, 0]),
                        'y': np.median(samples[:, 1])
                       }
            other_events['fixations'].append(this_fix)
        return other_events


    def showMeTheWay(self, positions_x, positions_y, figname, savedir, example_events=10, example_window_dur=2, use_filter=False, make_plots=True):
        '''
        example_events is number of randomly selected events of each type to plot
        example_window_dur is in seconds
        '''
        vel_x, vel_y = self.get_smoothedVelocity(positions_x), self.get_smoothedVelocity(positions_y)
        denoised_x, denoised_y = self.get_denoisedPositions(positions_x[0], vel_x), self.get_denoisedPositions(positions_y[0], vel_y)
        noise_x, noise_y = self.get_noise(vel_x), self.get_noise(vel_y)
        if self.verbose: print('noise x: ', noise_x, ' noise y: ', noise_y)
        if use_filter:
            velx_f = savgol_filter(vel_x, 27, 3, deriv=0, delta=1.0, axis=- 1, mode='constant', cval=0.0) #17, 3
            vely_f = savgol_filter(vel_y, 27, 3, deriv=0, delta=1.0, axis=- 1, mode='constant', cval=0.0)
        if use_filter:
            in_saccade = self.in_saccade(velx_f, vely_f, noise_x, noise_y)
        else:
            in_saccade = self.in_saccade(vel_x, vel_y, noise_x, noise_y)
        pre_events = self.presegment_events(in_saccade, denoised_x, denoised_y)
        events = self.segment_most(pre_events, denoised_x, denoised_y, vel_x, vel_y)
        events = self.segment_fixations(pre_events, events, len(denoised_x), denoised_x, denoised_y)


        if make_plots:
            fig = plot_summary(plt.figure(figsize=(17, 12)), events, len(positions_x), self.samplingRate)
            fig.suptitle('%s - %dmin %d s'%(figname, (len(positions_x)/self.samplingRate)/60, (len(positions_x)/self.samplingRate)%60))
            fig.savefig(os.path.join(savedir, figname+'.pdf'), bbox_inches='tight')

        if example_events and make_plots:
            for i in range(example_events):
                for ev_type in list(events.keys()):
                    n = np.random.choice(len(events[ev_type]))
                    if use_filter:
                        fig = plot_event(events, ev_type, n, vel_x, vel_y, denoised_x, denoised_y, velocities_x_filtered=velx_f, velocities_y_filtered=vely_f)
                    else:
                        fig = plot_event(events, ev_type, n, vel_x, vel_y, denoised_x, denoised_y)
                    fig.savefig(os.path.join(savedir, '%s_%i_%s.pdf'%(ev_type, n, figname)), bbox_inches='tight')
        if example_window_dur and make_plots: # here first samples could make it random
            inWinEvents = extractEventsInTimeWindow(events, 0, int(example_window_dur*self.samplingRate))
            len(inWinEvents)
            f, ax = plt.subplots(figsize=(8, 4))
            ax = plot_events_in_timewindow(ax, inWinEvents, 0, int(example_window_dur*self.samplingRate), denoised_x, denoised_y)
            f.savefig(os.path.join(savedir, 'eventsInWin_%d-%d_%s.pdf'%(0, example_window_dur, figname)), bbox_inches='tight')

        if use_filter:
            return events, denoised_x, denoised_y, vel_x, vel_y, velx_f, vely_f
        else:
            return events, denoised_x, denoised_y, vel_x, vel_y


