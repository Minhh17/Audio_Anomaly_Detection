from core.Preprocessing import Feature_extractor
import tensorflow as tf
import os
from pydub import AudioSegment
from pydub.utils import make_chunks
from helper.utils import read_file_name
from tqdm import tqdm
import numpy as np
from shutil import copy

class Dataloader(Feature_extractor):
    def __init__(self, cfg):
        super().__init__(
            type=cfg.PREPROCESS.TYPE, segment_len=cfg.PREPROCESS.SEGMENT_LEN, \
            audio_len=cfg.PREPROCESS.AUDIO_LEN, sample_per_file=cfg.PREPROCESS.SAMPLE_PER_FILE, \
            window_time=cfg.PREPROCESS.GAMMA.WINDOW_TIME, hop_time=cfg.PREPROCESS.GAMMA.HOP_TIME, \
            channels=cfg.PREPROCESS.GAMMA.CHANNELS, f_min=cfg.PREPROCESS.GAMMA.F_MIN, \
            sr=cfg.PREPROCESS.MEL.SR, nfft=cfg.PREPROCESS.MEL.NFFT, n_mel_band=cfg.PREPROCESS.MEL.N_BANDS
        )

        # some paths to data directories
        self.src_data_dir = {
            'normal':   cfg.DATASET.PATH.NORMAL,
            'anomaly':  cfg.DATASET.PATH.ANOMALY,
        }
        self.test_data_dir = cfg.DATASET.PATH.TEST
        
        self.base_tfrecord_list = cfg.DATASET.PATH.TFRECORDS
        self.target_tfrecord_list = cfg.TRANSFER_LEARNING.TFRECORDS
        self.tfrecord_dir = {
            'train':    os.path.join(self.base_tfrecord_list[0], 'train'),
            'val':      os.path.join(self.base_tfrecord_list[0], 'val'),
            'test':     os.path.join(self.base_tfrecord_list [0], 'test'),
            'normal_tl':  os.path.join(self.base_tfrecord_list[0], 'train'),
            'anomaly_tl': os.path.join(self.base_tfrecord_list[0], 'anomaly'),
        }
        self.stat_path = self.base_tfrecord_list[0]
        self.anomaly_tfrecord_dir = os.path.join(self.base_tfrecord_list[0], 'anomaly')
        os.makedirs(self.anomaly_tfrecord_dir, exist_ok=True)

        self.train_data_ratio = cfg.DATASET.RATIO.TRAIN
        self.test_data_ratio = cfg.DATASET.RATIO.TEST

        # parameters for creating dataloader
        self.batch_size = cfg.DATASET.DATALOADER.BATCH_SIZE
        self.shuffle = cfg.DATASET.DATALOADER.SHUFFLE

        self._init_feature_description()

    def _init_feature_description(self):
        self.feature_description = {
            'feature':  tf.io.FixedLenFeature([], tf.string),
            'label':    tf.io.FixedLenFeature([], tf.string),
            'idx':      tf.io.FixedLenFeature([], tf.string),
        }

    def _check_npz(self):
        return 'npz' in os.listdir(self.src_data_dir['normal'])[0]

    def _check_directories(self):
        for key, dir in self.src_data_dir.items():
            if dir == '':
                raise ValueError(f'Parameter src specifying data directory for {key} is not provided')
            elif not os.path.isdir(dir):
                raise ValueError(f'Folder {dir} does not exist.')
            else:
                print(f'Getting data from {dir}')

        # making sure the self.tfrecord_dir is existed
        for _, dst_dir in self.tfrecord_dir.items():
            os.makedirs(dst_dir, exist_ok=True)

    def _check_for_test(self, file_list, file_nums):
        '''
            Check if DATASET.PATH.TEST parameter is assigned
            If not, part of normal training data will be used for testing
        '''
        train_idx = int(self.train_data_ratio*file_nums)
        test_idx = int(self.test_data_ratio*file_nums)
        if self.test_data_dir:
            print(f"Getting test data from {self.test_data_dir}")
            data_dict = {
                'train':file_list[:train_idx+test_idx],
                'normal_test': read_file_name(self.test_data_dir),
                'val': file_list[train_idx+test_idx:],
            }
        else:
            data_dict = {
                'train':file_list[:train_idx],
                'normal_test': file_list[train_idx:train_idx+test_idx],
                'val': file_list[train_idx+test_idx:],
            }

        return data_dict

    def create_tfrecord(self):
        self.impl_func = {
            'npz': self._create_tfrecord_from_npz,
            'wav': self._create_tfrecord_from_wav,
        }['npz' if self._check_npz() else 'wav']
        self._check_directories()
        self.impl_func()

    def _bytes_feature(self, value):
        """Returns a bytes_list from a string / byte."""
        # first serialize the input
        value = tf.io.serialize_tensor(value)
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _create_tfrecord_from_wav(self):
        time_per_sample = self.segment_len*1000  # time is processed in millisecond
        rate = self.audio_len//self.segment_len
        # read all files in the directory
        test_id_holder = 0 # a work around to save both anomaly test sample and normal test sample to the same directory
        for data_type, src_dir in self.src_data_dir.items():
            file_list = read_file_name(src_dir)
            file_nums = len(file_list)
            if not file_nums:
                print(f'{src_dir} directory is empty.')
                return None

            if 'normal' in data_type:
                data_dict = self._check_for_test(file_list, file_nums)
                label = 0
            else:
                data_dict = {
                    'anomaly_test': file_list
                }
                label = 1

            for data_part, data_list in data_dict.items():
                part = data_part.split('_')[-1]
                record_idx = test_id_holder if 'test' in data_part else 0
                # number of fully-written records
                num = (len(data_list)*rate)//self.sample_per_file + record_idx
                # number of sample in half-written records
                remainder = (len(data_list)*rate) % self.sample_per_file  
                # a counter for the number of samples has been proccessed
                sample_counter = 0
                idx_list = []
                feature_list = []

                for file in tqdm(data_list, desc=f'Extracting {data_part} features'):
                    audio = AudioSegment.from_file(file, "wav")
                    chunks = make_chunks(audio, time_per_sample)

                    for index, item in enumerate(chunks):
                        feature = self.feat_extr_func[self.type](item)
                        # print(feature)
                        feature_list.append(feature)
                        name = file.split('/')[-1]
                        idx_list.append(name[:-4]+'_'+str(index))
                        sample_counter += 1
                        # saving back features to .tfrecord format
                        if  (sample_counter % self.sample_per_file == 0) or \
                            (record_idx == num and sample_counter == remainder) or \
                            (num == 0 and sample_counter == remainder):
                            file_path = os.path.join(self.tfrecord_dir[part], f'data_{record_idx:08}.tfrecord')
                            with tf.io.TFRecordWriter(file_path) as writer:
                                for feature, id in zip(feature_list, idx_list):    
                                    temp = tf.train.Example(features=tf.train.Features(
                                        feature={
                                            'feature':  self._bytes_feature(feature.astype(np.float32)),
                                            'label':    self._bytes_feature(label),
                                            'idx':      self._bytes_feature(id),
                                        }
                                    )).SerializeToString()
                                    writer.write(temp)
                            # copy anomaly tfrecord files to another directory
                            if 'anomaly' in data_part:
                                copy(file_path, os.path.join(self.anomaly_tfrecord_dir, os.path.split(file_path)[-1]))
                            # update and reset counter
                            record_idx += 1
                            sample_counter = 0
                            feature_list = []
                            idx_list = []

                # hold the index value of tfrecord if data_part is test
                if 'test' in data_part:      
                    test_id_holder = record_idx

    def _create_tfrecord_from_npz(self, use_anomaly=False):
        '''
            Load already extracted features from .npz file and save them to .tfrecord files.
            First, we load all samples into respected list.
            Then create tfrecord accordingly
        '''
        # Define a specialized function here
        def save_tfrecord_from_nparray(data, data_type, idx_list=None, anomaly_set=False):
            label = 1 if anomaly_set else 0
            idx_list = ['unknown']*data.shape[0] if not idx_list else idx_list

            record_num = np.ceil(data.shape[0]/self.sample_per_file).astype(np.int16)
            remainder = data.shape[0]%self.sample_per_file
            remainder_after_remove_faulty = remainder

            for i in range(record_num):
                file_path = os.path.join(self.tfrecord_dir[data_type], f'data_{i:08}.tfrecord')
                sample_num = self.sample_per_file if i != (record_num-1) else remainder_after_remove_faulty
                start_id = i*self.sample_per_file
                if sample_num <= 0:
                    break
                with tf.io.TFRecordWriter(file_path) as writer:
                    for sample_id in range(sample_num):
                        if sample_id >= remainder_after_remove_faulty:
                            break
                        sample = data[start_id + sample_id]
                        if sample.shape == (32,32):    
                            temp = tf.train.Example(features=tf.train.Features(
                                feature={
                                    'feature':  self._bytes_feature(sample.astype(np.float32)),
                                    'label':    self._bytes_feature(label),
                                    'idx':      self._bytes_feature(idx_list[start_id + sample_id]),
                                }
                            )).SerializeToString()
                            writer.write(temp)
                        else:
                            print('Detecting faulty sample')
                            remainder_after_remove_faulty -= 1
                # copy anomaly tfrecord files to another directory
                if anomaly_set:
                    copy(file_path, os.path.join(self.anomaly_tfrecord_dir, os.path.split(file_path)[-1]))

        # Load all sample and ids (from .json file)
        dict_of_samples_list = {
            'normal': [],
            'anomaly': [],
        }
        
        for part, directory in self.src_data_dir.items():
            file_list = read_file_name(directory)
            for file in file_list:
                dict_of_samples_list[part].append(np.load(file)['arr_0'])

        # Divide normal data into train, val and test set using indices
        sample_array = np.concatenate(dict_of_samples_list['normal'], axis=0)
        sample_num = sample_array.shape[0]
        train_idx = int(self.train_data_ratio*sample_num)
        test_idx = int(self.test_data_ratio*sample_num)
        train_set = sample_array[:train_idx]
        test_normal = sample_array[train_idx:train_idx+test_idx]
        val_set = sample_array[train_idx+test_idx:]

        # Using the specialized function that will save tfrecord
        for data_set, name in zip([train_set, test_normal, val_set], ['train', 'test', 'val']):
            save_tfrecord_from_nparray(data=data_set, data_type=name, anomaly_set=False)

        # Do the same for anomaly set if use_anomaly=True
        if use_anomaly:
            sample_array = np.concatenate(dict_of_samples_list['anomaly'], axis=0)
            save_tfrecord_from_nparray(data=sample_array, data_type='test', anomaly_set=True)

    def _check_idx_exist(self):
        pass

    def _parse_function(self, input_proto):
        # Parse the input `tf.train.Example` proto using the dictionary feature_description.
        parsed_sample = tf.io.parse_single_example(input_proto, self.feature_description)
        return  (tf.io.parse_tensor(parsed_sample['feature'], tf.float32), \
                tf.io.parse_tensor(parsed_sample['label'], tf.int32), parsed_sample['idx'])

    def create_dataloader(self, data_part, batch_size=None):
        abs_path = lambda x: os.path.join(self.tfrecord_dir[data_part], x)
        tfrecords_list = list(map(abs_path, os.listdir(self.tfrecord_dir[data_part])))

        dataset = tf.data.TFRecordDataset(tfrecords_list)
        parsed_dataset = dataset.map(self._parse_function)
        bs = batch_size if batch_size else self.batch_size
        if self.shuffle:
            parsed_dataset = parsed_dataset.shuffle(buffer_size=1000)

        return parsed_dataset.batch(batch_size=bs).prefetch(tf.data.AUTOTUNE)

    def create_tl_dataloader(self, data_part, batch_size=None):
        tfrecords_list = []
        record_list = self.base_tfrecord_list if data_part != 'test' else self.target_tfrecord_list
        for dir in record_list:
            tfrecords_list += read_file_name(os.path.join(dir, data_part))
        
        return self.create_dataloader_from_files(tfrecords_list, batch_size)

    def create_dataloader_from_files(self, list_of_files, batch_size=None):
        dataset = tf.data.TFRecordDataset(list_of_files)
        parsed_dataset = dataset.map(self._parse_function)
        bs = batch_size if batch_size else self.batch_size
        if self.shuffle:
            parsed_dataset = parsed_dataset.shuffle(buffer_size=1000)

        return parsed_dataset.batch(batch_size=bs).prefetch(tf.data.AUTOTUNE)

    def accumulate_stat(self):
        train_data = self.create_dataloader('train', 1)

        MIN = np.iinfo(np.int16).max
        MAX = np.iinfo(np.int16).min


        for feature, _, _ in tqdm(train_data, desc='Accumulating statistics'):
            temp1 = tf.reduce_min(feature)
            temp2 = tf.reduce_max(feature)
            MIN = MIN if MIN<temp1 else temp1
            MAX = MAX if MAX>temp2 else temp2

        with open(os.path.join(self.stat_path, 'stats.npz'), 'wb') as file:
            np.savez(file, max=MAX, min=MIN)
