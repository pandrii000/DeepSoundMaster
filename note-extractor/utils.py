import librosa
import numpy as np
import scipy
from python_speech_features import mfcc
from midiutil import MIDIFile


def extract_features(
        signal,
        n_features=512,
        feature_name='fft',
        sample_rate=44100,
        frame_duration=1024):
    
    if feature_name == 'clean':
        padded_length = int(np.ceil(len(signal) / frame_duration)) * frame_duration
        x = np.zeros(padded_length)
        x[:len(signal)] = signal
        return x.reshape(-1, frame_duration)
    
    elif feature_name == 'mfcc':
        return mfcc(signal       = signal,
                    samplerate   = sample_rate,
                    winlen       = frame_duration / sample_rate,
                    winstep      = frame_duration / sample_rate,
                    numcep       = n_features,
                    nfilt        = 40,
                    nfft         = frame_duration,
                    lowfreq      = 0,
                    highfreq     = 22050,
                    preemph      = 0.95,
                    ceplifter    = 0,
                    appendEnergy = True)
    
    elif feature_name == 'fft':
        padded_length = int(np.ceil(len(signal) / frame_duration)) * frame_duration
        x = np.zeros(padded_length)
        x[:len(signal)] = signal
        x = x.reshape(-1, frame_duration)
        fft = np.absolute(np.fft.fft(x))
        fft = fft[:, :fft.shape[1] // 2]
        assert fft.shape[1] == n_features, (fft.shape, n_features)
        return fft

    elif feature_name == 'spectrogram':
         padded_length = int(np.ceil(len(signal) / frame_duration)) * frame_duration
         x = np.zeros(padded_length)
         f, t, Sxx = scipy.signal.spectrogram(
             x               = x,
             fs              = sample_rate,
             window          = 'hamming',
             nperseg         = frame_duration,
             noverlap        = None,
             detrend         = False,
             return_onesided = True,
             scaling         = 'spectrum',
             axis            = -1,
             mode            = 'magnitude')
         return Sxx

    else:
        raise Exception()


def extract_midi(h5dataset, dim, frame_duration=1024):
    # only 128 midi notes exist
    y = np.zeros((dim, 128), dtype='bool')
    
    for row in h5dataset:
        frame_start = row['start_time'] // frame_duration
        frame_end   = row['end_time'] // frame_duration + 1
        
        y[frame_start: frame_end + 1, row['note_id']] = 1
    
    return y


def readfile(filepath):
    signal, sr = librosa.load(filepath, sr=None)
    if sr != 44100:
        signal = librosa.resample(signal, sr, 44100)
    return signal


def union_notes(prediction):
    for i in range(prediction.shape[0] - 1, 1, -1):
        i_notes = np.where(prediction[i] != 0)[0]
        iprev_notes = np.where(prediction[i-1] != 0)[0]
        common_notes = np.array(list(set(i_notes).intersection(set(iprev_notes))))
        if len(common_notes) > 0:
            prediction[i, common_notes] = 0
            prediction[i-1, common_notes] += 1


def export_to_midi(prediction, output_path):
    midifile = MIDIFile(numTracks=1)
    track     = 0
    channel   = 0
    volume    = 80
    time      = 0
    trackname = 'restored'
    bpm       = 44100 * 60 // 1024
    midifile.addTrackName(track, time, trackname)
    midifile.addTempo(track, time, bpm)

    for i, y in enumerate(prediction):
        pitches = np.where(y >= 1)[0]
        for pitch in pitches:
            duration = y[pitch]
            midifile.addNote(track, channel, pitch, i, duration, volume)   

    with open(output_path, "wb") as f:
        midifile.writeFile(f)
