# import 
import torch
import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer, Wav2Vec2Processor
import torch
import torch.nn.functional as F
max_length = 2048
path='/workspace/omkar_projects/PyTorch-YOLOv3/checkpoint-3500'
# processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
processor = Wav2Vec2Processor.from_pretrained(path)
# load the tokenizer and model
tokenizer = Wav2Vec2Tokenizer.from_pretrained(path)
model = Wav2Vec2ForCTC.from_pretrained(path).to('cuda')

# load the audio data (use your own wav file here!)
input_audio, sr = librosa.load('/workspace/omkar_projects/PyTorch-YOLOv3/00000da4b2da194ac31f6d1466d2ceb4823bdc263efa81004ee89464.wav', sr=16000)

# tokenize
input_values = processor(input_audio, return_tensors="pt", padding="longest").input_values.to('cuda')
print(input_values.shape)
# retrieve logits
logits = model(input_values).logits
print(logits.shape)
sequence_length = logits.shape[1]
padding_length = max_length - sequence_length
if padding_length > 0:
    padding = (0,padding_length)
    logits = F.pad(logits, (0, 0, padding[0], padding[1]), "constant", 0)
print(logits.shape)
# take argmax and decode
predicted_ids = torch.argmax(logits, dim=-1)
transcription = tokenizer.batch_decode(predicted_ids)

# print the output
print(transcription)