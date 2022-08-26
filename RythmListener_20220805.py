#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  4 10:23:16 2022

@author: scola
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib.colors import Normalize
import librosa
import librosa.display
import IPython.display


ry_intervals = {
        '8ve' : {'name' : 'octave', 'fraction' : 1/2, 'color' : 'slategrey', 'name_EN' : '8th'},
        '7e min' : {'name' : 'septième mineure', 'fraction' : 4/7, 'color' : '#9467bd', 'name_EN' : 'min 7th'},
        '6te' : {'name' : 'sixte', 'fraction' : 3/5, 'color' : '#ff7f0e', 'name_EN' : '6th'},
        '6te min' : {'name' : 'sixte mineure', 'fraction' : 5/8, 'color' : 'brown', 'name_EN' : 'min 6th'},
        '5te' : {'name' : 'quinte', 'fraction' : 2/3, 'color' : '#d62728', 'name_EN' : '5th'},
        '5te dim' : {'name' : 'quinte diminuée', 'fraction' : 5/7 ,'color' : 'gold', 'name_EN' : 'dim 5th'},
        '4te' : {'name' : 'quarte', 'fraction' : 3/4, 'color' : '#2ca02c', 'name_EN' : '4th'},         
        '3ce' : {'name' : 'tierce', 'fraction' : 4/5, 'color' : '#1f77b4', 'name_EN' : '3rd'},                          
        '3ce min' : {'name' : 'tierce mineure', 'fraction' : 5/6, 'color' : 'purple', 'name_EN' : 'min 3rd'},                          
        '2de' : {'name' : 'seconde', 'fraction' : 7/8, 'color' : '#8c564b', 'name_EN' : '2nd'},
        '1son' : {'name' : 'seconde', 'fraction' : 1, 'color' : 'dimgray', 'name_EN' : '1'}
        }


class SIGNAL :
    wavecolor = 'Indigo' #'Blue'
    secondcol = 'Purple'
    spec_cmap = 'cividis' 
    spec_cmap = 'Oranges'
    spec_cmap = 'YlOrBr'
    spec_cmap = 'viridis' 
#    spec_cmap = 'BuPu'
    figure_size = (12,6)
    
    def __init__ (self, filename, frame_size = 256, tightness = 100, beat_frame_offset = 0):
        self.file = filename.split ('/')[-1][:-4]
        signal, sample_rate = librosa.load(filename)
        self.signal = signal
        self.samplerate = sample_rate
        self.n_sample = len (signal)
        self.time = np.linspace (0, self.n_sample / self.samplerate, self.n_sample)
        self.frame_size = frame_size
        self.tightness = tightness
        self.tempo_factor = 1
        self.beat_frame_shift = 0
        self.delta = 1
        self.hide_unitones = True
        self.show_rynterval_names = False
        self.unitone_deviation = 0.05
        self.specgram_cscale = (-66, -28)
        
    
    def track_tempo (self, tightness = 100):
        self.tightness = tightness
        tempo, beat_frame = librosa.beat.beat_track(y=self.signal,
                                                    sr=self.samplerate,
                                                    hop_length = self.frame_size,
                                                    tightness = self.tightness)
        self.tempo = tempo
        self.beat_frame = beat_frame
        durations = ()
        for i in range (len (self.beat_frame) - 1):
            durations += (self.beat_frame [i+1] - self.beat_frame [i]),
        self.tempo_stability = 100 - np.std (np.array (durations)) / np.mean (np.array (durations)) * 100
        return None
    
    def plotwaveform (self, ax):
        ax.plot (self.time, self.signal, color = self.wavecolor, lw = 0.5)
        ax.set_xlabel ('Time (s)')
        ax.set_ylabel ('Amplitude (arb. unit)')
        ax.set_title (self.file)
        return None

    def display_tempo (self, ax):
        yext = ax.get_ylim ()
        ax.vlines (self.beat_frame * self.frame_size/ self.samplerate, yext [0], yext [1], color = 'k', alpha = 0.6)
        return None    

    def playback (self):
        IPython.display.Audio (self.signal, rate=self.samplerate)
        return None
    
    def track_enveloppe (self):
        self.onset_enveloppe = librosa.onset.onset_strength (y=self.signal, sr=self.samplerate, hop_length = self.frame_size)
        self.n_frame = len (self.onset_enveloppe)
        self.time_onset = np.linspace (0, self.n_frame * self.frame_size/ self.samplerate, self.n_frame)
        return None
    
    def display_enveloppe (self, ax):
        ax.plot (self.time_onset, self.onset_enveloppe, color = self.wavecolor)    
        ax.set_ylabel ('signal energy enveloppe (arb. unit)')
        return None
    
    def track_onsets (self, delta = 0):
        self.onset_frames = librosa.util.peak_pick(self.onset_enveloppe, pre_max=7, post_max=7, pre_avg=7, post_avg=7, delta = delta, wait=1)
        return None
    
    def display_onsets (self, ax, extent = [0, 10], color = 'k', alpha = 0.4):
        yext = ax.get_ylim ()
        if extent == [0, 10]:
            extent = yext
        ax.vlines (np.array (self.onset_frames) * self.frame_size / self.samplerate,
                   extent [0], extent [1], color = color, alpha = alpha, lw = 0.5)
        ax.set_ylim (0, yext [1])
        return None

    def calculate_onset_consonnace (self):
        rychords = ()
        time_rychords = ()
        for i in range (len (self.beat_frame)-1):
            unit = self.beat_frame [i+1] - self.beat_frame [i]
            onsets = (np.array (self.onset_frames) - self.beat_frame [i]) / unit
            rychord = ()
            for onset in onsets:
                if 0 <= onset < 1:
                    rychord += onset,
            rychords += rychord,
            time_rychords += self.beat_frame [i],
        self.rychords = rychords
        self.time_rychords = time_rychords
        return None
        
    def rhythm_chords_hist (self, ax2, n_bins = 100):        
        # build chords 
        rychords_hist = ()
        for i in range (len (self.time_rychords)):
            for j in range (len (self.rychords [i])):
                rychords_hist += self.rychords [i][j],
        # display histogram
        self.rychords_hist = rychords_hist
        hist_data = ax2.hist (self.rychords_hist, bins = n_bins, color = self.wavecolor)
        ax2.set_xlim (0,1)
        ax2.xaxis.set_minor_locator(AutoMinorLocator())
        #ax2.set_tick_params (axis = 'x', which = 'minor')
        ymax = ax2.get_ylim ()[1]
        ax2.set_xlabel ('relative onsets position in time unit')
        
        # display background intervals
        
        for interval in ry_intervals:
            ax2.vlines (ry_intervals [interval]['fraction'],
                        0, 0.98 * ymax, color = ry_intervals [interval]['color'], 
                        ls = '-', lw = 1.2, alpha = 0.8)
            ax2.text (ry_intervals [interval]['fraction'] - 1e-2, ymax * 1.05, interval,
                      color = ry_intervals [interval]['color'], rotation = 45)
            
            ax2.vlines (ry_intervals [interval]['fraction'] / 2,
                        0, 0.98 * ymax, color = ry_intervals [interval]['color'], 
                        ls = '-', lw = 1.2, alpha = 0.4)
            ax2.vlines (ry_intervals [interval]['fraction'] / 4,
                        0, 0.98 * ymax, color = ry_intervals [interval]['color'], 
                        ls = '-', lw = 1.2, alpha = 0.2)    
        return hist_data


    def display_rhythm_chords (self, ax1, hide_unitones = True, eps = 0.05, show_rynterval_names = False):        
        # build chords 
        rychords_hist = ()
        for i in range (len (self.time_rychords)):
            ax1.vlines (self.time_rychords [i] * self.frame_size / self.samplerate, 0, 1,
                        color = 'k', ls = 'dotted', alpha = 0.1)
            for j in range (len (self.rychords [i])):
                if hide_unitones and eps <= self.rychords [i][j] <= 1 - eps:
                    ax1.scatter (self.time_rychords [i] * self.frame_size / self.samplerate,
                                 self.rychords [i][j], marker = 'o', color = self.wavecolor)
                    rychords_hist += self.rychords [i][j],
        ax1.set_ylim (0,1)
        ax1.set_ylabel ('relative onsets position in time unit')
        ax1.set_xlabel ('time (s)')
        xmin, xmax = ax1.get_xlim()
        # display histogram
        self.rychords_hist = rychords_hist
        
        # display background intervals       
        for interval in ry_intervals:
            ax1.hlines (ry_intervals [interval]['fraction'],
                        0, xmax, color = ry_intervals [interval]['color'], 
                        ls = '-', lw = 1.2, alpha = 0.8)
            if show_rynterval_names: 
                ax1.text (xmax * 1.07, ry_intervals [interval]['fraction'] - 1e-2, interval,
                          color = ry_intervals [interval]['color'], rotation = 0)
            ax1.hlines (ry_intervals [interval]['fraction'] / 2,
                        0, xmax, color = ry_intervals [interval]['color'], 
                        ls = '-', lw = 1.2, alpha = 0.4)
            ax1.hlines (ry_intervals [interval]['fraction'] / 4,
                        0, xmax, color = ry_intervals [interval]['color'], 
                        ls = '-', lw = 1.2, alpha = 0.2)    
        return None
        
    
    def plot_specgram (self, ax, title = False, vmin = -66, vmax = -28):
        output = ax.specgram (self.signal,
                     xextent = (self.time [0], self.time [-1]),
                     NFFT = self.frame_size,
                     pad_to = self.frame_size*8,
                     Fs = self.samplerate,
                     scale = 'dB',
                     norm = Normalize (vmin=vmin, vmax = vmax),
                     cmap = self.spec_cmap)
        ax.set_xlabel ('Time (s)')
        ax.set_ylabel ('Frequency (Hz)')
        if title : 
            ax.set_title (self.file)
        ymin, ymax = ax.get_ylim ()
        ax.set_yscale ('log')
        ax.set_ylim (40, ymax)
        return None
    
    
    def burn_clicks (self):
        self.clicks = librosa.clicks(times = self.beat_frame * self.frame_size / self.samplerate, sr=self.samplerate, length = self.n_sample)
        return None
        
        
    def playback_clicked_track (self):
        IPython.display.Audio(self.signal + self.clicks, rate=self.samplerate)
        return None
    
    def display_spec_and_onset (self, figsize, vmin = -66, vmax = -28):
        fig, (ax1, ax2) = plt.subplots (nrows = 2, ncols = 1, figsize = (figsize [0], 2 * figsize [1]))
        self.plot_specgram (ax1, vmin = vmin, vmax = vmax)
        yext = ax1.get_ylim ()
        xext = ax1.get_xlim ()
        self.display_onsets (ax1, extent = [yext [1]/500, yext [1]], color = 'gray', alpha = 1)
        self.display_enveloppe (ax2)
        self.display_onsets (ax2)
        self.display_tempo (ax2)
        ax1.set_xlabel ('')
        ax2.set_xlabel ('time (s)')
        ax2.set_xlim (xext)
        return None
    
    def half_tempo (self):
        self.beat_frame = self.beat_frame [0::2]
        self.tempo /= 2
        return None
    
    def double_tempo (self):
        new_beat_frame = [self.beat_frame [0]]
        print (self.beat_frame)
        for i in range (len (self.beat_frame)-1):
            new_beat_frame.append (int (0.5 * self.beat_frame [i+1] + 0.5 * self.beat_frame [i]))
            new_beat_frame.append (self.beat_frame [i+1])
        self.beat_frame = np.array (new_beat_frame)
        self.tempo *= 2
        print (self.beat_frame)
        return None
    
    def shift_beat_frame (self, shift = 0.5):
        self.beat_frame += int (60 / self.tempo * shift * self.samplerate / self.frame_size)
        return None
    
    def analyse (self, delta = 0):
        figure_size = (12,6)
        # piece.playback ()    
        
        # get track tempo
        self.track_tempo ()
        
        # display wave form
        fig, ax = plt.subplots (figsize = figure_size)
        self.plotwaveform (ax)
        self.display_tempo (ax)
            
        # add tempo clicks to the original track
        self.burn_clicks ()
        self.playback_clicked_track ()
        
        fig, ax5 = plt.subplots (figsize = figure_size)
        self.plot_specgram (ax5)
        
        # track enveloppe
        self.track_enveloppe ()
        
        # display enveloppe
        fig, ax2 = plt.subplots (figsize = figure_size)
        self.display_enveloppe (ax2)
         
        # track onset
        self.track_onsets (self.delta)
        self.display_onsets (ax2)
        self.display_tempo (ax2)
    
        # calculate onset consonnance
        self.calculate_onset_consonnace ()
    
        # display rhythm chords
        fig, (ax3, ax4) = plt.subplots (nrows = 1, ncols = 2, figsize = figure_size)
        hist_data = self.display_rhythm_chords (ax3, ax4)
        
        
        self.display_spec_and_onset (figsize = figure_size)
        return hist_data
    
    def triple_analyse (self, ax, DEBUG = False):
        ax1, ax2, ax3 = ax
                
        # get track tempo
        self.track_tempo (tightness = self.tightness)
        
        # impose time unit
        n = int (np.log (self.tempo_factor) / np.log (2))
        for i in range (n):
            if n < 0:
                self.half_tempo ()
            if n > 0:
                self.double_tempo ()
        
        # track enveloppe
        self.track_enveloppe ()

        # track onset
        self.track_onsets (self.delta)

        # calculate onset consonnance
        self.calculate_onset_consonnace ()
    
        # display wave form
        self.plot_specgram (ax1,
                            vmin = self.specgram_cscale [0],
                            vmax = self.specgram_cscale [1])
        self.display_enveloppe (ax2)
        self.display_onsets (ax2)
        self.display_tempo (ax2)
        self.display_rhythm_chords (ax3,
                                    hide_unitones = self.hide_unitones,
                                    eps = self.unitone_deviation,
                                    show_rynterval_names = self.show_rynterval_names)
        xlim = ax1.get_xlim ()
        ax2.set_xlim (xlim)
        ax3.set_xlim (xlim)
        ylim = ax3.get_ylim ()
        plt.yscale ('log')
        ax3.set_ylim ((0.1, 1))
        plt.axes (ax1)
        plt.xticks ([])
        plt.xlabel ('')
        plt.axes (ax2)
        plt.xticks ([])
        plt.axes (ax1, ymargin = 0.)
        
        return None


    def make_tempogram (self):
        self.tempogram = librosa.feature.tempogram (onset_envelope = self.onset_enveloppe,
                                                    sr = self.samplerate,
                                                    hop_length = self.frame_size)
        return None
            
    def display_tempogram (self, ax):
        print ('coucou')
        return None
    
def main ():
    print ('RythmListener\nExplore fine rhythms with a little help from my CPU...')
    path2source = '/home/scola/RECHERCHE/Collab_Lasayah/sources/AUDIO/youtube-dl/extraits/'
    filename = 'Gnawa_Home_songs-Bouyandi_extrait-1.wav'
    filename = 'JS_Bach-Fugue_Cmin.wav'
    filename = 'DAngelo_Method_Man_Red_Man-Left_and_Right_intro.wav'
    path2source = '/home/joseph/MEGAsync/Collab_LaSayah/RythmListener/sources/AUDIO/youtube-dl/extraits/'
    path2source = '/home/joseph/Collab_Lasayah/RythmListener/sources/AUDIO/extraits/'
    filename = 'Lindigo-Buffalony_D1.flac'
#    filename = 'Lindigo-Maloya_power.flac'
    # import signal
    piece = SIGNAL (path2source + filename, tightness = 200)

    #fig, ax = plt.subplots ()
#    piece.track_tempo ()
#    piece.shift_beat_frame (-0.02)
#    piece.double_tempo ()
#    piece.plotwaveform (ax)
#    piece.plot_specgram (ax, vmin = -66, vmax = -28)
#    piece.display_tempo (ax)    
#    piece.track_enveloppe ()
#    piece.track_onsets (delta = 1)
#    piece.display_enveloppe (ax)
#    piece.display_onsets (ax)
#    piece.display_tempo (ax)
#    piece.display_spec_and_onset (figsize = piece.figure_size, vmin = -66, vmax = -12)
#    piece.calculate_onset_consonnace ()

    figure_size = (8,18)
    fig, ax = plt.subplots (nrows = 3, ncols = 1, figsize = figure_size)
    piece.triple_analyse (ax)   
    fig, ax = plt.subplots ()
    piece.rhythm_chords_hist (ax)
#    
#    fig, ax = plt.subplots ()
#    piece.tempogram = librosa.feature.tempogram (y = None, 
#                                                 onset_envelope = piece.onset_enveloppe,
#                                                 hop_length = piece.frame_size)    
#    librosa.display.specshow (piece.tempogram, cmap = 'magma', ax = ax,
#                              sr = 1 / piece.frame_size/ piece.samplerate,
#                              x_axis = 'time', y_axis = 'log')
    plt.show ()
    return None

if __name__ == '__main__' :
    main ()