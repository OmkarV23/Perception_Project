import pyttsx3

"""
sudo apt install git
python3 -m pip install git+https://github.com/nateshmbhat/pyttsx3
"""

engine = pyttsx3.init()
# voices = engine.getProperty('voices')
# engine.setProperty('voice', voices[1].id)

engine.save_to_file("find bottle", "./recordings/pyttsx3_find_bottle.wav")
engine.save_to_file("find cup", "./recordings/pyttsx3_find_cup.wav")
engine.save_to_file("find sports ball",
                    "./recordings/pyttsx3_find_sports_ball.wav")
engine.save_to_file("grab bottle", "./recordings/pyttsx3_grab_bottle.wav")
engine.save_to_file("grab cup", "./recordings/pyttsx3_grab_cup.wav")
engine.save_to_file("grab sports ball",
                    "./recordings/pyttsx3_grab_sports_ball.wav")
engine.save_to_file("lift bottle", "./recordings/pyttsx3_lift_bottle.wav")
engine.save_to_file("lift cup", "./recordings/pyttsx3_lift_cup.wav")
engine.save_to_file("lift sports ball",
                    "./recordings/pyttsx3_lift_sports_ball.wav")
engine.save_to_file(
    "pick-up bottle", "./recordings/pyttsx3_pick_up_bottle.wav")
engine.save_to_file("pick-up cup", "./recordings/pyttsx3_pick_up_cup.wav")
engine.save_to_file("pick-up sports ball",
                    "./recordings/pyttsx3_pick_up_sports_ball.wav")
engine.runAndWait()

# get the list of all the available voices
voices = engine.getProperty('voices')
for voice in voices:
    print(voice.id)
