from os import path
from pydub import AudioSegment

# files
src = "./recordings/Antoni_cup.mp3"
dst = "./recordings/Antoni_cup.wav"

# convert wav to mp3
sound = AudioSegment.from_mp3(src)
sound.export(dst, format="wav")
