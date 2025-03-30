from os.path import join, isdir, isfile
from os import mkdir, listdir
from pydub import AudioSegment
from pydub.utils import make_chunks
from argparse import ArgumentParser

parser = ArgumentParser(description='Split WAV file(s) into 2s segments')
parser.add_argument('-s', '--source', help='source file or folder with WAV files', required=True)
parser.add_argument('-d', '--destination', help='destination folder for split files', required=True)
args = parser.parse_args()

def segmentation(src_file, dst, length=2):
    try:
        audio = AudioSegment.from_file(src_file, format="wav")
        duration_ms = len(audio)
        print(f"Processing {src_file}: {duration_ms} ms ({duration_ms / 1000:.2f} seconds)")
    except Exception as e:
        print(f"Error loading {src_file}: {e}")
        return 0

    time_per_file = length * 1000  # 2s = 2000ms
    chunks = make_chunks(audio, time_per_file)
    print(f"Total chunks generated: {len(chunks)}")

    file_name = src_file.split('/')[-1][:-4]
    exported_count = 0

    for index, item in enumerate(chunks):
        chunk_duration = len(item)
        chunk_name = f"{file_name}_{index}.wav"
        print(f"Chunk {index}: {chunk_duration} ms - Exporting {chunk_name}")
        item.export(join(dst, chunk_name), format="wav")
        exported_count += 1

    print(f"Exported {exported_count} chunks from {src_file}")
    return exported_count

def process_input(source, destination):
    if not isdir(destination):
        print(f"Creating destination folder: {destination}")
        mkdir(destination)

    total_chunks = 0
    if isfile(source):
        # Trường hợp đầu vào là file
        if not source.lower().endswith('.wav'):
            print(f"Error: {source} is not a WAV file")
            return
        print(f"Source is a file: {source}")
        total_chunks = segmentation(source, destination)
    elif isdir(source):
        # Trường hợp đầu vào là folder
        wav_files = [f for f in listdir(source) if f.lower().endswith('.wav')]
        print(f"Source is a folder: {source}")
        print(f"Found {len(wav_files)} WAV files in {source}")
        for wav_file in wav_files:
            src_path = join(source, wav_file)
            total_chunks += segmentation(src_path, destination)
    else:
        print(f"Error: {source} is neither a valid file nor a directory")
        return

    print(f"Total chunks exported: {total_chunks}")

if __name__ == "__main__":
    source = args.source
    destination = args.destination
    print(f"Source: {source}")
    print(f"Destination folder: {destination}")
    process_input(source, destination)
    print('Completed')