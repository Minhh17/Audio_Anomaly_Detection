from helper.utils import read_file_name, extract_mbe
from os.path import join, isdir, split
from os import makedirs
from pydub import AudioSegment
from pydub.utils import make_chunks
import numpy as np
from gammatone import gtgram
import json
from tqdm import tqdm


class Feature_extractor():
    def __init__(self, src=None, dst=None, mode='from_file', type=None,\
        segment_len=None, audio_len=None, sample_per_file=200, \
        window_time=None, hop_time=None, channels=None, f_min=None, \
        sr=16000, nfft=None, n_mel_band=32):
        # audio directory and output directory
        self.src = src
        self.dst = dst
        
        # usage of feature extractor: to prepare data from audio files (from_file) or from streaming data (real_time)
        self.mode = mode
        self.type = type

        # length of each audio file
        self.audio_len = audio_len
        self.segment_len = segment_len
        self.sample_per_file = sample_per_file 

        # parameters for gamma features
        self.window_time = window_time
        self.hop_time = hop_time
        self.channels = channels
        self.f_min = f_min

        # parameters for mel features
        self.sr = sr
        self.nfft = nfft
        self.n_mel_band = n_mel_band
        
        # a dictionary to save back all ids
        self.data = {}

        # a dict of implemented function
        self.feat_extr_func = {
            'gamma': self._get_gamma_feature,
            'mel': self._get_mel_feature,
        }

    def extract_feature(self, type='gamma'):
        self._check_directories()
        self._check_type(type)
        self._extract_feature_from_file()

    def extract_feature_rt(self, audio, type='gamma'):
        self._check_type(type)
        self._extract_feature_from_stream(audio)

    def _check_type(self, type):
        if not type in ['gamma', 'mel']:
            raise ValueError(f'{type} is not one of the implemented methods. Please choose between (gamma) and (mel)')
        else:
            if self.type:
                print(f'type parameter is now overwritten with {type}')
            self.type = type

    def _check_directories(self):
        if not self.src:
            raise ValueError('Parameter src specifying audio directory is not provided')
        elif not isdir(self.src):
            raise ValueError(f'Folder {self.src} does not exist.')
        else:
            print(f'Getting audio files from {self.src}')
        # making sure the self.dst is existed
        makedirs(self.dst, exist_ok=True)

    def _extract_feature_from_file(self):
        time_per_sample = self.segment_len*1000  # time is processed in millisecond

        # read all files in the directory
        file_list = read_file_name(self.src)
        if len(file_list) == 0:
            return None

        rate = self.audio_len//self.segment_len
        file_nums = len(file_list)
        num = (file_nums*rate)//self.sample_per_file
        remainder = (file_nums*rate) % self.sample_per_file
        i = 0
        j = 0
        idx = []
        feature_list = []

        for file in tqdm(file_list, desc=f'Extracting {type} features from {self.src}'):
            audio = AudioSegment.from_file(file, "wav")
            # print('audiosegment')
            chunks = make_chunks(audio, time_per_sample)
            for index, item in enumerate(chunks):
                feature = self.impl_func[self.type](item)
                feature_list.append(feature)
                name = file.split('/')[-1]
                idx.append(name[:-4]+'_'+str(index))
                j += 1
                # saving back features to .npz file format
                if (j % self.sample_per_file == 0) or (i == num and j == remainder) or (num == 0 and j == remainder):    # noqa: E501
                    np.savez_compressed(join(self.dst, str(i)), np.array(feature_list))
                    i += 1
                    j = 0
                    feature_list = []

        name_list = self.dst.split('/')
        category = name_list[-2]+'_'+name_list[-1]
        print(f'Saving {category}')
        self.data[category] = idx

    def _extract_feature_from_stream(self, audio):
        return self.impl_func[self.type](audio)

    def _get_gamma_feature(self, input):
        # .reshape((-1, item.channels)) if needed
        chunk = np.array(input.get_array_of_samples(), dtype=np.float32)/(2*(8*input.sample_width-1)+1)
        # print(chunk.shape)
        gtg = gtgram.gtgram(chunk, input.frame_rate, self.window_time, \
            self.hop_time, self.channels, self.f_min)
        return np.flipud(20 * np.log10(gtg))

    def _get_mel_feature(self, input):
        # .reshape((-1, item.channels)) if needed
        chunk = np.array(input.get_array_of_samples(), dtype=np.float32)/(2*(8*input.sample_width-1)+1)
        # print(chunk.shape)
        print(self.sr, self.nfft, self.n_mel_band)
        return extract_mbe(chunk, self.sr, self.nfft, self.n_mel_band)

    # use this method to save back id dictionary only after finishing extract features for all part of the dataset
    def save_id(self):
        path = split(self.dst)
        part = self.dst.split('/')[-1]
        name = f'{part}_idx.json'
        with open(join(path, name), 'w') as file:
            json.dump(self.data, file)
        self.data = {}
