#! /usr/bin/env python3

from __future__ import division

import os, sys
import argparse
import tqdm
import librosa
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from transformers import Wav2Vec2Processor

from models import load_model
from utils.logger import Logger
from utils.utils import to_cpu, load_classes, print_environment_info, provide_determinism, worker_seed_set
from utils.datasets import ListDataset
from utils.augmentations import AUGMENTATION_TRANSFORMS
from utils.parse_config import parse_data_config, parse_model_config
from utils.loss import compute_loss
from validation import _evaluate, _create_validation_data_loader
#from utils.transforms import DEFAULT_TRANSFORMS
from terminaltables import AsciiTable

from torchsummary import summary


def _create_data_loader(data, image_dir,
                        annotation_dir, audio_dir, processor, 
                        batch_size, img_size, n_cpu, 
                        multiscale_training=False):
    """Creates a DataLoader for training.

    :param img_path: Path to file containing all paths to training images.
    :type img_path: str
    :param batch_size: Size of each image batch
    :type batch_size: int
    :param img_size: Size of each image dimension for yolo
    :type img_size: int
    :param n_cpu: Number of cpu threads to use during batch generation
    :type n_cpu: int
    :param multiscale_training: Scale images to different sizes randomly
    :type multiscale_training: bool
    :return: Returns DataLoader
    :rtype: DataLoader
    """
    dataset = ListDataset(
        data, image_dir,
        annotation_dir, audio_dir, processor,
        img_size=img_size,
        multiscale=multiscale_training,
        transform=AUGMENTATION_TRANSFORMS)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        # num_workers=n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
        worker_init_fn=worker_seed_set)
    return dataloader


def run():
    print_environment_info()
    parser = argparse.ArgumentParser(description="Trains the YOLO model.")
    parser.add_argument("-m", "--model", type=str, default="/workspace/omkar_projects/PyTorch-YOLOv3/config/yolov3.cfg", help="Path to model definition file (.cfg)")
    parser.add_argument("-d", "--data", type=str, default="config/coco.data", help="Path to data config file (.data)")
    parser.add_argument("-e", "--epochs", type=int, default=300, help="Number of epochs")
    parser.add_argument("-v", "--verbose", action='store_true', help="Makes the training more verbose")
    parser.add_argument("--n_cpu", type=int, default=1, help="Number of cpu threads to use during batch generation")
    parser.add_argument("--image", type=str, help="Path to checkpoint file (.weights or .pth). Starts training from checkpoint model")
    parser.add_argument("--audio", type=str, help="Fine tuned Wave2vec2 checkpoints")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="Interval of epochs between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=1, help="Interval of epochs between evaluations on validation set")
    parser.add_argument("--multiscale_training", action="store_true", help="Allow multi-scale training")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="Evaluation: IOU threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.1, help="Evaluation: Object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.5, help="Evaluation: IOU threshold for non-maximum suppression")
    parser.add_argument("--logdir", type=str, default="logs", help="Directory for training log files (e.g. for TensorBoard)")
    parser.add_argument("--seed", type=int, default=-1, help="Makes results reproducable. Set -1 to disable.")
    args = parser.parse_args()
    print(f"Command line arguments: {args}")

    if args.seed != -1:
        provide_determinism(args.seed)

    logger = Logger(args.logdir)  # Tensorboard logger

    # Create output directories if missing
    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Get data configuration
    data_config = parse_data_config(args.data)
    
    image_dir = data_config["image_dir"]
    annotation_dir = data_config["annotation_dir"]
    audio_dir = data_config["audio_dir"]
    train_data = data_config["train"]
    valid_data = data_config["valid"]
    class_names = load_classes(data_config["names"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = 'cpu'

    # ############
    # Create model
    # ############

    #wave2_vec_path = '/workspace/omkar_projects/PyTorch-YOLOv3/new_ckpt_w2v'
    processor = Wav2Vec2Processor.from_pretrained(args.audio)
    model = load_model(args.model,args.audio,args.image).to(device)
    
    hyperparams = parse_model_config(args.model).pop(0)
    hyperparams.update({
        'batch': int(hyperparams['batch']),
        'subdivisions': int(hyperparams['subdivisions']),
        'width': int(hyperparams['width']),
        'height': int(hyperparams['height']),
        'channels': int(hyperparams['channels']),
        'optimizer': hyperparams.get('optimizer'),
        'momentum': float(hyperparams['momentum']),
        'decay': float(hyperparams['decay']),
        'learning_rate': float(hyperparams['learning_rate']),
        'burn_in': int(hyperparams['burn_in']),
        'max_batches': int(hyperparams['max_batches']),
        'policy': hyperparams['policy'],
        'lr_steps': list(zip(map(int,   hyperparams["steps"].split(",")),
                             map(float, hyperparams["scales"].split(","))))
    })


    # if args.verbose:
    #     summary(model, input_size=[(3, hyperparams['height'], hyperparams['height']), (109712)], device='cpu')

    mini_batch_size = hyperparams['batch'] // hyperparams['subdivisions']
    # #################
    # Create Dataloader
    # #################

    #Load training dataloader
    dataloader = _create_data_loader(
        train_data,
        image_dir,
        annotation_dir,
        audio_dir,
        processor,
        mini_batch_size,
        hyperparams['height'],
        args.n_cpu,
        args.multiscale_training)

    # Load validation dataloader
    validation_dataloader = _create_data_loader(
        valid_data,
        image_dir,
        annotation_dir,
        audio_dir,
        processor,
        mini_batch_size,
        hyperparams['height'],
        args.n_cpu)


    # ################
    # Create optimizer
    # ################

    params = [p for p in model.parameters() if p.requires_grad]
    
    if (hyperparams['optimizer'] in [None, "adam"]):
        optimizer = optim.Adam(
            params,
            lr=hyperparams['learning_rate'],
            weight_decay=hyperparams['decay'],
        )
    elif (hyperparams['optimizer'] == "sgd"):
        optimizer = optim.SGD(
            params,
            lr=hyperparams['learning_rate'],
            weight_decay=hyperparams['decay'],
            momentum=hyperparams['momentum'])
    else:
        print("Unknown optimizer. Please choose between (adam, sgd).")

    # skip epoch zero, because then the calculations for when to evaluate/checkpoint makes more intuitive sense
    # e.g. when you stop after 30 epochs and evaluate every 10 epochs then the evaluations happen after: 10,20,30
    # instead of: 0, 10, 20
    for epoch in range(1, args.epochs+1):

        print("\n---- Training Model ----")

        model.train()  # Set model to training mode

        for batch_i, (_, _, imgs, targets, audio_logits) in enumerate(tqdm.tqdm(dataloader, desc=f"Training Epoch {epoch}")):
            batches_done = len(dataloader) * epoch + batch_i
            imgs = imgs.to(device, non_blocking=True)
            audio_logits = torch.stack(audio_logits).to(device, non_blocking=True).squeeze(1)
            targets = targets.to(device)

            outputs = model(imgs, audio_logits)

            loss, loss_components = compute_loss(outputs, targets, model)

            loss.backward()

            ###############
            # Run optimizer
            ###############

            if batches_done % hyperparams['subdivisions'] == 0:
                # Adapt learning rate
                # Get learning rate defined in cfg
                lr = hyperparams['learning_rate']
                if batches_done < hyperparams['burn_in']:
                    # Burn in
                    lr *= (batches_done / hyperparams['burn_in'])
                else:
                    # Set and parse the learning rate to the steps defined in the cfg
                    for threshold, value in hyperparams['lr_steps']:
                        if batches_done > threshold:
                            lr *= value
                # Log the learning rate
                logger.scalar_summary("train/learning_rate", lr, batches_done)
                # Set learning rate
                for g in optimizer.param_groups:
                    g['lr'] = lr

                # Run optimizer
                optimizer.step()
                # Reset gradients
                optimizer.zero_grad()

            # ############
            # Log progress
            # ############
            if args.verbose:
                print(AsciiTable(
                    [
                        ["Type", "Value"],
                        ["IoU loss", float(loss_components[0])],
                        ["Object loss", float(loss_components[1])],
                        ["Class loss", float(loss_components[2])],
                        ["Loss", float(loss_components[3])],
                        ["Batch loss", to_cpu(loss).item()],
                    ]).table)

            # Tensorboard logging
            tensorboard_log = [
                ("train/iou_loss", float(loss_components[0])),
                ("train/obj_loss", float(loss_components[1])),
                ("train/class_loss", float(loss_components[2])),
                ("train/loss", to_cpu(loss).item())]
            logger.list_of_scalars_summary(tensorboard_log, batches_done)

            model.seen += imgs.size(0)

        # #############
        # Save progress
        # #############

        # Save model to checkpoint file
        if epoch % args.checkpoint_interval == 0:
            checkpoint_path = f"checkpoints/yolov3_ckpt_{epoch}.pth"
            print(f"---- Saving checkpoint to: '{checkpoint_path}' ----")
            torch.save(model.state_dict(), checkpoint_path)

        # ########
        # Evaluate
        # ########

        if epoch % args.evaluation_interval == 0:
            print("\n---- Evaluating Model ----")
            # Evaluate the model on the validation set
            metrics_output = _evaluate(
                model,
                validation_dataloader,
                class_names,
                img_size=hyperparams['height'],
                iou_thres=args.iou_thres,
                conf_thres=args.conf_thres,
                nms_thres=args.nms_thres,
                verbose=args.verbose
            )

            if metrics_output is not None:
                precision, recall, AP, f1, ap_class = metrics_output
                evaluation_metrics = [
                    ("validation/precision", precision.mean()),
                    ("validation/recall", recall.mean()),
                    ("validation/mAP", AP.mean()),
                    ("validation/f1", f1.mean())]
                logger.list_of_scalars_summary(evaluation_metrics, epoch)
        sys.exit(0)

if __name__ == "__main__":
    run()
