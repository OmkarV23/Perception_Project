from pydub import AudioSegment
# install ffmpeg to avoid the warnings
# sudo apt install -y ffmpeg

# enter the path to the two WAV audio files
sound1 = AudioSegment.from_file("./2023-03-02-18-29-40.wav")
sound2 = AudioSegment.from_file("./2023-03-02-18-30-14.wav")


def merge_two_audios():
    """
    Merge the two audio files supplied as input
    """
    print("Audio files being merged...")
    sound3 = sound1.append(sound2, crossfade=0)
    sound3.export("./merged_sound.wav", format="wav")
    print("Audio files merged!")


merge_two_audios()
