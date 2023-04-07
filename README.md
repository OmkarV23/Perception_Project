# Perception_Project

## :ear: + :eyes: using :hugs:

Added SRGAN script to rescale image 4X.

Download weights darknet53 from weights/download_weights.sh
comment yolov3 and yolov3-tiny weights before downloading, if not commented.

Download the Wave2vec 2.0 fine tuned checkpoints for this task. [link](https://drive.google.com/drive/folders/114Ydozgz_mON0KqnQr2dNDrTXD_1Vgt2?usp=share_link)

Added custom torchsummary to include custom :hugs: modules.

Run single pass as:

`python3 python3 train.py --data <location of maping file> --model <location of the cfg file> --image <location of darknet53 C binary> --audio <location of finetuned wave2vec 2.o checkpoints>`

