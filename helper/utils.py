import wave
import librosa
import numpy as np
import os

# a function goes through directoty and gives back list of files
def read_file_name(path):

    file_list = []
    if os.path.isdir(path):
        file_list = os.listdir(path)
        file_list = [os.path.join(path, file) for file in file_list]
    else:
        return 0

    return file_list


def load_audio(filename, mono=True, fs=44100):
    """
    Load audio file into numpy array
    ----------
    Parameters:
    filename:  str
        Path to audio file

    mono : bool
    In case of multi-channel audio, channels are averaged into single channel.
    (Default value=True)

    fs : int > 0 [scalar]
    Target sample rate, if input audio does not fulfil this, audio is resampled
    (Default value=44100)
    -------
    Returns:
    audio_data : numpy.ndarray [shape=(channel, signal_length)]
        Audio

    sample_rate : integer
        Sample rate
    """

    file_base, file_extension = os.path.splitext(filename)
    if file_extension == '.wav':
        _audio_file = wave.open(filename)

        # Audio info
        sample_rate = _audio_file.getframerate()
        sample_width = _audio_file.getsampwidth()
        number_of_channels = _audio_file.getnchannels()
        number_of_frames = _audio_file.getnframes()

        # Read raw bytes
        data = _audio_file.readframes(number_of_frames)
        _audio_file.close()

        # Convert bytes based on sample_width
        num_samples, remainder = divmod(len(data),
                                        sample_width * number_of_channels)
        if remainder > 0:
            raise ValueError('The length of data is not a multiple of sample \
                                size * number of channels.')
        if sample_width > 4:
            raise ValueError('Sample size cannot be bigger than 4 bytes.')

        if sample_width == 3:
            # 24 bit audio
            a = np.empty((num_samples, number_of_channels, 4), dtype=np.uint8)
            raw_bytes = np.frombuffer(data, dtype=np.uint8)
            a[:, :, :sample_width] = raw_bytes.reshape(-1, number_of_channels, sample_width)   # noqa: E501
            a[:, :, sample_width:] = (a[:, :, sample_width - 1:sample_width] >> 7) * 255   # noqa: E501
            audio_data = a.view('<i4').reshape(a.shape[:-1]).T
        else:
            # 8 bit samples are stored as unsigned ints; others as signed ints.
            dt_char = 'u' if sample_width == 1 else 'i'
            a = np.frombuffer(data, dtype='<%s%d' % (dt_char, sample_width))
            audio_data = a.reshape(-1, number_of_channels).T

        if mono:
            # Down-mix audio
            audio_data = np.mean(audio_data, axis=0)

        # Convert int values into float
        audio_data = audio_data / float(2 ** (sample_width * 8 - 1) + 1)

        # Resample
        # if fs != sample_rate:
        #     audio_data = librosa.core.resample(audio_data, sample_rate, fs)
        #     sample_rate = fs

        return audio_data, sample_rate
    return None, None


def extract_mbe(y, sr, n_fft, n_mel_band):
    # Compute the mel spectrogram
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=n_fft//2, n_mels=n_mel_band, power=1.0)
    return mel_spec
