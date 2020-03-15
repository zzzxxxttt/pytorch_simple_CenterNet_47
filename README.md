# Pytorch simple CenterNet-47

If you are looking for another CenterNet, try [this](https://github.com/zzzxxxttt/pytorch_simple_CenterNet_45)!

This repository is a simple pytorch implementation of [CenterNet: Keypoint Triplets for Object Detection](https://arxiv.org/abs/1904.08189), some of the code is taken from the [official implementation](https://github.com/Duankaiwen/CenterNet).
As the name says, this version is **simple** and **easy to read**, all the complicated parts (dataloader, hourglass, training loop, etc) are all rewrote in a simpler way.    
By the way the support of **nn.parallel.DistributedDataParallel** is also added, so this implementation trains considerably faster than the official code.

Enjoy!     
 
## Requirements:
- python>=3.5
- pytorch>=0.4.1(>=1.1.0 for DistributedDataParallel training)
- tensorboardX(optional)

## Getting Started
1. Disable cudnn batch normalization.
Open `torch/nn/functional.py` and find the line with `torch.batch_norm` and replace the `torch.backends.cudnn.enabled` with `False`.

2. Clone this repo:
    ```
    CenterNet_ROOT=/path/to/clone/CornerNet
    git clone https://github.com/zzzxxxttt/pytorch_simple_CenterNet47 $CenterNet_ROOT
    ```

3. Install COCOAPI (the cocoapi in this repo is modified to work with python3):
    ```
    cd $CenterNet_ROOT/lib/cocoapi/PythonAPI
    make
    python setup.py install --user
    ```

4. Compile corner pooling.
    If you are using pytorch 0.4.1, rename ```$CenterNet_ROOT/lib/cpool_old``` to ```$CenterNet_ROOT/lib/cpool```, otherwise rename ```$CenterNet_ROOT/lib/cpool_new``` to ```$CenterNet_ROOT/lib/cpool```.
    ```
    cd $CenterNet_ROOT/lib/cpool
    python setup.py install --user
    ```

5. Compile NMS.
    ```
    cd $CenterNet_ROOT/lib/nms
    make
    ```

6. For COCO training, Download [COCO dataset](http://cocodataset.org/#download) and put ```annotations```, ```train2017```, ```val2017```, ```test2017``` (or create symlinks) into ```$CenterNet_ROOT/data/coco```

## Train 
### COCO

#### multi GPU using nn.parallel.DistributedDataParallel
```
python -m torch.distributed.launch --nproc_per_node NUM_GPUS train.py --dist \
        --log_name coco_hg_511_ddp \
        --dataset coco \
        --arch large_hourglass \
        --lr 5e-4 \
        --lr_step 90,120 \
        --batch_size 48 \
        --num_epochs 200 \
        --num_workers 2
```

## Evaluate
### COCO
```
python test.py --log_name coco_hg_511_dp \
               --dataset coco \
               --arch large_hourglass

# flip test
python test.py --log_name coco_hg_511_dp \
               --dataset coco \
               --arch large_hourglass \
               --test_flip

# multi scale test
python test.py --log_name coco_hg_511_dp \
               --dataset coco \
               --arch large_hourglass \
               --test_flip \
               --test_scales 0.5,0.75,1,1.25,1.5
```

## Results:

### COCO:
Model|Training image size|mAP
:---:|:---:|:---:
Hourglass-52 (DDP)|511|39.5/41.9/43.6
Hourglass-104 (DDP)|511|42.9/45.0/46.9



