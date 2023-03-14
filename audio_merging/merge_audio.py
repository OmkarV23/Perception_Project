from pydub import AudioSegment

# install ffmpeg to avoid the warnings
# sudo apt install -y ffmpeg

# enter the path to the verb WAV audio files
find = AudioSegment.from_file("./recordings/Sam_find.wav")
grab = AudioSegment.from_file("./recordings/Sam_grab.wav")
lift = AudioSegment.from_file("./recordings/Sam_lift.wav")
pick_up = AudioSegment.from_file("./recordings/Sam_pick_up.wav")

# enter the path to the subject WAV audio files
cup = AudioSegment.from_file("./recordings/Sam_cup.wav")
bottle = AudioSegment.from_file("./recordings/Sam_bottle.wav")
sports_ball = AudioSegment.from_file("./recordings/Sam_sports_ball.wav")


def merge_sentence():
    """
    Merge the two audio files supplied as input
    """
    print("Audio files being merged...")
    # find
    sentence = find.append(cup, crossfade=400)
    sentence.export("./recordings/Sam_find_cup.wav", format="wav")
    sentence = find.append(bottle, crossfade=400)
    sentence.export("./recordings/Sam_find_bottle.wav", format="wav")
    sentence = find.append(sports_ball, crossfade=400)
    sentence.export("./recordings/Sam_find_sports_ball.wav", format="wav")
    # grab
    sentence = grab.append(cup, crossfade=400)
    sentence.export("./recordings/Sam_grab_cup.wav", format="wav")
    sentence = grab.append(bottle, crossfade=400)
    sentence.export("./recordings/Sam_grab_bottle.wav", format="wav")
    sentence = grab.append(sports_ball, crossfade=400)
    sentence.export("./recordings/Sam_grab_sports_ball.wav", format="wav")
    # lift
    sentence = lift.append(cup, crossfade=350)
    sentence.export("./recordings/Sam_lift_cup.wav", format="wav")
    sentence = lift.append(bottle, crossfade=350)
    sentence.export("./recordings/Sam_lift_bottle.wav", format="wav")
    sentence = lift.append(sports_ball, crossfade=350)
    sentence.export("./recordings/Sam_lift_sports_ball.wav", format="wav")
    # pick-up
    sentence = pick_up.append(cup, crossfade=400)
    sentence.export("./recordings/Sam_pick_up_cup.wav", format="wav")
    sentence = pick_up.append(bottle, crossfade=400)
    sentence.export("./recordings/Sam_pick_up_bottle.wav", format="wav")
    sentence = pick_up.append(sports_ball, crossfade=400)
    sentence.export("./recordings/Sam_pick_up_sports_ball.wav", format="wav")
    print("Audio files merged!")


merge_sentence()
