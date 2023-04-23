import onnxruntime
import librosa
import numpy as np
from postprocess import postprocess
# from stitch import stitch
import detect
import sounddevice as sd
from scipy.io.wavfile import write
from transformers import Wav2Vec2Processor
import cv2

sd.default.device = 'sof-hda-dsp: - (hw:2,7)'
sampling_rate = 16000
duration = 4  # seconds
baseline = 7
alpha = 117.104

global input_audio

audio_path = '/workspace/omkar_projects/PyTorch-YOLOv3/pytorchyolo/audio_sample.wav'
classes = detect.load_classes('/workspace/omkar_projects/PyTorch-YOLOv3/data/coco.names')
wave2_vec_path = "/workspace/omkar_projects/PyTorch-YOLOv3/fine_tuned_wave2vec_english"
processor = Wav2Vec2Processor.from_pretrained(wave2_vec_path)
print('Loading model')
model = onnxruntime.InferenceSession("model_xlsr.onnx", providers=['CUDAExecutionProvider'])
print('Loaded model')

cv_file = cv2.FileStorage()
cv_file.open('stereoMap.xml', cv2.FileStorage_READ)

stereoMapL_x = cv_file.getNode('stereoMapL_x').mat()
stereoMapL_y = cv_file.getNode('stereoMapL_y').mat()
stereoMapR_x = cv_file.getNode('stereoMapR_x').mat()
stereoMapR_y = cv_file.getNode('stereoMapR_y').mat()


def undistortRectify(frameR, frameL):

    # Undistort and rectify images
    undistortedL= cv2.remap(frameL, stereoMapL_x, stereoMapL_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    undistortedR= cv2.remap(frameR, stereoMapR_x, stereoMapR_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)

    return undistortedR, undistortedL

def find_depth(right_point, left_point, frame_right, frame_left, baseline, alpha):

    # # CONVERT FOCAL LENGTH f FROM [mm] TO [pixel]:
    # height_right, width_right, depth_right = frame_right.shape
    # height_left, width_left, depth_left = frame_left.shape

    # f_pixel = ((width_right+width_left)/2 * 0.5) / np.tan(alpha * 0.5 * np.pi/180)
    f_pixel = (820.50061035 + 814.63525391 + 815.39221191 + 812.79290771) / 4

    x_right = right_point[0]
    x_left = left_point[0]

    # CALCULATE THE DISPARITY:
    disparity = -x_left+x_right      #Displacement between left and right frames [pixels]

    # CALCULATE DEPTH z:
    zDepth = (baseline*f_pixel)/disparity             #Depth in [cm]

    return abs(zDepth)

def preprocess(image):
    image = cv2.resize(image, (416, 416))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)
    return image

def draw(center, depth, frame, detections, img_size, output_path, classes, frame_name):
    colour = (255,0,0)
    for (x1, y1, x2, y2, conf, cls_pred),det_id in zip(detections,range(center.shape[0])):
        # print(center[det_id,:])
        x1, y1, x2, y2 = round(x1.item()), round(y1.item()), round(x2.item()), round(y2.item())
        # xc, yc = x1 + ((x2-x1) / 2), y1 + ((y2-y1) / 2)
        print(f"\t+ Label: {classes[int(cls_pred)]} | Confidence: {conf.item():0.4f}")

        # draw center of bounding box
        # print(centers)
        frame = cv2.circle(frame, (round(center[det_id,:][0]), round(center[det_id,:][1])), 2, colour, 2)
        frame = cv2.rectangle(frame,(x1,y1),(x2,y2),colour,2)
        frame = cv2.putText(frame,classes[int(cls_pred)],(x1,y1),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
        frame = cv2.putText(frame,str(round(depth,2))+'cm',(x1,y1+20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
    return frame

cap1 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(2)
cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

audio, sr = librosa.load(audio_path, sr=sampling_rate)
input_audio = processor(audio, return_tensors="pt", padding="longest", sampling_rate=sampling_rate).input_values.numpy()


# input_audio = np.random.rand(1,16000).astype(np.float32)
while cap1.isOpened() and cap2.isOpened():
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()          
    if ret1 and ret2:
        frame1, frame2 = undistortRectify(frame1, frame2)
        input_image = np.concatenate((preprocess(frame1), preprocess(frame2)), axis=0)
        input_image = input_image.astype(np.float32) / 255.0
        outputs = model.run(['Bounding_boxes','audio_embeddings'], {
            "image_input": input_image,
            "audio_input": input_audio
        })
        detections = postprocess(frame1, outputs, processor, classes)
        if len(detections[0])!=0 and len(detections[1])!=0:

            detections[0] = np.array([i.tolist() for i in sorted(detections[0], key=lambda x: x[0])])
            detections[1] = np.array([i.tolist() for i in sorted(detections[1], key=lambda x: x[0])])

            cam1_c = np.concatenate([(detections[0][:,0] + ((detections[0][:,2] - detections[0][:,0]) /2.0))[:,None], 
                                    (detections[0][:,1] + ((detections[0][:,3] - detections[0][:,1]) /2.0))[:,None]], axis=1)     
            cam2_c = np.concatenate([(detections[1][:,0] + ((detections[1][:,2] - detections[1][:,0]) /2.0))[:,None],
                                    (detections[1][:,1] + ((detections[1][:,3] - detections[1][:,1]) /2.0))[:,None]], axis=1)      

            for i,j in zip(cam1_c,cam2_c):
                depth = find_depth(i, j, frame1, frame2, baseline, alpha)

            frame1 = draw(cam1_c,depth,frame1,detections[0],416,".",classes,'frame1')
            frame2 = draw(cam2_c,depth,frame2,detections[1],416,".",classes,'frame2')
           
        cv2.imshow("frame1", frame1)
        cv2.imshow("frame2", frame2)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

