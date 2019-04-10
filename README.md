# A Pruned Version for Lightweight OpenPose

<p align="center">
  <img src="data/preview.jpg" />
</p>

## Table of Contents

* [Requirements](#requirements)
* [Prerequisites](#prerequisites)
* [Training](#training)
* [Validation](#validation)
* [Pruned model](#pruned_model)
* [Fine-tuned model](#fine-tuned_model)
* [Unpruned pre-trained model](#pre-trained-model)

## Requirements

* Ubuntu 16.04
* Python 3.6
* PyTorch 0.4.1 (should also work with 1.0, but not tested)

## Prerequisites

1. Download COCO 2017 dataset: [http://cocodataset.org/#download](http://cocodataset.org/#download) (train, val, annotations) and unpack it to `<COCO_HOME>` folder.
2. Install requirements `pip install -r requirements.txt`

## Training

1. Fine-tune the pruned model. Run `CUDA_VISIBLE_DEVICES=<DEVICES_ID> python train_prune.py --train-images-folder ./coco/train2017/ --prepared-train-labels prepared_train_annotation.pkl --val-labels val_subset.json --val-images-folder ./coco/val2017/ --checkpoint-path ./pruned_models/<CHECKPOINT> --num-refinement-stages 3 --experiment-name <NAME> --weights-only`

## Validation

1. For training-time synchronous validation. Run `CUDA_VISIBLE_DEVICES=<DEVICES_ID> python val_per_epoch.py`
2. Validation for a specific checkpoint. Run `python val_prune_oneepoch.py --labels <COCO_HOME>/annotations/person_keypoints_val2017.json --images-folder <COCO_HOME>/val2017 --checkpoint-path <CHECKPOINT>`

## Pruned model <a name="pruned_model"/>

We provide two pruned models with different compression rate: ./pruned_models/0.3.pth.tar (reduce 15.92% flops) and ./pruned_models/0.8.tar.pth (reduce 25.6% flops).

## Fine-tuned model <a name="fine-tuned_model"/>

The model fine-tuned from the pruned model `./pruned_models/0.3.pth.tar` is available in `./fine-tuned_models/`. 

## Unpruned pre-trained model <a name="pre-trained-model"/>

The model expects normalized image (mean=[128, 128, 128], scale=[1/256, 1/256, 1/256]) in planar BGR format.
Pre-trained on COCO model is available at: ./pre-trained_models/checkpoint_iter_370000.pth.tar, it has 40% of AP on COCO validation set (38.6% of AP on the val *subset*).

