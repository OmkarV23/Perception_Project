from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import glob
import random
import os
import warnings
import librosa
import numpy as np
import pandas as pd
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


class ImageFolder(Dataset):
    def __init__(self, folder_path, transform=None):
        self.files = sorted(glob.glob("%s/*.*" % folder_path))
        self.transform = transform

    def __getitem__(self, index):

        img_path = self.files[index % len(self.files)]
        img = np.array(
            Image.open(img_path).convert('RGB'),
            dtype=np.uint8)

        # Label Placeholder
        boxes = np.zeros((1, 5))

        # Apply transforms
        if self.transform:
            img, _ = self.transform((img, boxes))

        return img_path, img

    def __len__(self):
        return len(self.files)


class ListDataset(Dataset):
    def __init__(self, data_frame, image_dir,
                annotation_dir, audio_dir, processor,
                img_size=640, multiscale=True, transform=None):
 
        self.data_frame = pd.read_csv(data_frame)
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.audio_dir = audio_dir
        self.processor = processor

        self.img_size = img_size
        self.max_objects = 100
        self.multiscale = multiscale
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0
        self.transform = transform

    def __getitem__(self, index):

        # ---------
        #  Image
        # ---------
        
        img_path = os.path.join(self.image_dir,self.data_frame.iloc[index].Images)
            # img_path = self.img_files[index % len(self.img_files)].rstrip()
        try:
            img = np.array(Image.open(img_path).convert('RGB'), dtype=np.uint8)
        except Exception:
            print(f"Could not read image '{img_path}'.")
            return

        # ---------
        #  Label
        # ---------
        
        label_path = os.path.join(self.annotation_dir,self.data_frame.iloc[index].Annotations)
            # label_path = self.label_files[index % len(self.img_files)].rstrip()
        try:
            # Ignore warning if file is empty
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                bb_targets = np.loadtxt(label_path).reshape(-1, 5)
        except Exception:
            print(f"Could not read label '{label_path}'.")
            return

        # -----------
        #  Transform
        # -----------
        if self.transform:
            try:
                img, bb_targets = self.transform((img, bb_targets))
            except Exception:
                print("Could not apply transform.")
                return
            
        # ---------
        #  Audio
        # ---------
        
        audio_path = os.path.join(self.audio_dir,self.data_frame.iloc[index].array_path)
            # audio_path = self.audio_files[index % len(self.img_files)].rstrip()
        try:
            # Ignore warning if file is empty
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                audio = np.load(audio_path)
                audio_logits = self.processor(audio,return_tensors ='pt',padding='longest',sampling_rate=22050).input_values.squeeze(0)
        except Exception:
            print(f"Could not read audio '{audio_path}'.")
            return

        return img_path, audio_path, img, bb_targets, audio_logits

    def collate_fn(self, batch):
        self.batch_count += 1

        # Drop invalid images
        batch = [data for data in batch if data is not None]

        image_paths, audio_paths, imgs, bb_targets, audio_logits = list(zip(*batch))

        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(
                range(self.min_size, self.max_size + 1, 32))

        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])

        # Add sample index to targets
        for i, boxes in enumerate(bb_targets):
            boxes[:, 0] = i
        bb_targets = torch.cat(bb_targets, 0)

        return image_paths, audio_paths, imgs, bb_targets, audio_logits

    def __len__(self):
        return len(self.data_frame)
