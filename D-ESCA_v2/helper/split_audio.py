from os.path import join, isdir
from os import mkdir, listdir
from pydub import AudioSegment
from pydub.utils import make_chunks


# a function goes through directoty and gives back list of files
def read_file_name(path):

    file_list = []
    if isdir(path):
        file_list = listdir(path)
        file_list = [join(path, file) for file in file_list]
    else:
        return 0

    return file_list

# a function segment the audio file to desired length in second
def segmentation(src, dst, length=10):
    # src is the source folder
    # dst is the destination folder

    time_per_file = length*1000  # time is processed in millisecond

    if not isdir(src):
        return 1

    print(src)
    file_list = read_file_name(src)
    print(len(file_list))

    print(dst)
    if not isdir(dst):
        mkdir(dst)

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


if __name__ == '__main__':
    # sources and destinations are folders that contain audio files
    # please specify the path to these folder in absolute path
    sources = []
    destinations = []
    duration = 2

    for src, dst in zip(sources, destinations):
      print(src)
      print(dst)
      segmentation(src, dst, duration)
    print('Completed')
