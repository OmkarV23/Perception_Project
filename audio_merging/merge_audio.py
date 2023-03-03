from pydub import AudioSegment

# install ffmpeg to avoid the warnings
# sudo apt install -y ffmpeg

# enter the path to the verb WAV audio files
find = AudioSegment.from_file("./find.wav")
grab = AudioSegment.from_file("./grab.wav")
lift = AudioSegment.from_file("./lift.wav")
pick_up = AudioSegment.from_file("./pick-up.wav")

# enter the path to the subject WAV audio files
cup = AudioSegment.from_file("./cup.wav")
bottle = AudioSegment.from_file("./bottle.wav")
sports_ball = AudioSegment.from_file("./sports_ball.wav")


def merge_sentence():
    """
    Merge the two audio files supplied as input
    """
    print("Audio files being merged...")
    # find
    sentence = find.append(cup, crossfade=800)
    sentence.export("./find_cup.wav", format="wav")
    sentence = find.append(bottle, crossfade=800)
    sentence.export("./find_bottle.wav", format="wav")
    sentence = find.append(sports_ball, crossfade=800)
    sentence.export("./find_sports_ball.wav", format="wav")
    # grab
    sentence = grab.append(cup, crossfade=800)
    sentence.export("./grab_cup.wav", format="wav")
    sentence = grab.append(bottle, crossfade=800)
    sentence.export("./grab_bottle.wav", format="wav")
    sentence = grab.append(sports_ball, crossfade=800)
    sentence.export("./grab_sports_ball.wav", format="wav")
    # lift
    sentence = lift.append(cup, crossfade=800)
    sentence.export("./lift_cup.wav", format="wav")
    sentence = lift.append(bottle, crossfade=800)
    sentence.export("./lift_bottle.wav", format="wav")
    sentence = lift.append(sports_ball, crossfade=800)
    sentence.export("./lift_sports_ball.wav", format="wav")
    # pick-up
    sentence = pick_up.append(cup, crossfade=800)
    sentence.export("./pick_up_cup.wav", format="wav")
    sentence = pick_up.append(bottle, crossfade=800)
    sentence.export("./pick_up_bottle.wav", format="wav")
    sentence = pick_up.append(sports_ball, crossfade=800)
    sentence.export("./pick_up_sports_ball.wav", format="wav")
    print("Audio files merged!")


merge_sentence()
