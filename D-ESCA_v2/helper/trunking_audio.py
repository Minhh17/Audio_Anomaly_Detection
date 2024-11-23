from os.path import join, isdir
from os import makedirs
from pydub import AudioSegment
from pydub.utils import make_chunks
from helper.utils import read_file_name

def segmentation(src, dst, length=10):
    '''
        Chunking .wav file into smaller segments of length
        ----------
        Parameters:
            src: the directory storing original .wav files
            dst: the directory storing segmented .wav files
            length: the disired length of the segment
    '''

    time_per_file = length*1000  # time is processed in millisecond

    if not isdir(src):
        raise Exception(f'Source directory is not correctly provided: {src}')

    file_list = read_file_name(src)
    print(f'Imported {len(file_list)} files from {src}')

    if not isdir(dst):
        makedirs(dst, exist_ok=True)

    for file in file_list:
        audio = AudioSegment.from_file(file, "wav")
        chunks = make_chunks(audio, time_per_file)
        file_name = file.split('/')[-1]

        # write chunks to dst
        for index, item in enumerate(chunks):
            chunk_name = file_name[:-4] + '_' + str(index) + '.wav'
            print("exporting ", chunk_name)
            item.export(join(dst, chunk_name), format='wav')

    return 0
