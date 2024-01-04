import os
import random
import argparse
import numpy as np

# multi gpu
import horovod.torch as hvd

import torch

from models.networks.ofa_resnets import OFAResNets
from models.modules.dynamic_op import DynamicSeparableConv2d
from ofa.utils.my_dataloader.my_random_resize_crop import MyRandomResizedCrop
from ofa.utils.run_config import DistributedImageNetRunConfig
from ofa.modules.distributed_run_manager import DistributedRunManager

parser = argparse.ArgumentParser()
# nas settings
parser.add_argument("--task", type=str, default="depth", choices=["kernel", "depth", "expand",])
parser.add_argument("--phase", type=int, default=1, choices=[1, 2]) # select phase 1 or 2 (depth and expand)

# genereal settings
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--n_classes", type=int, default=1000)
parser.add_argument("--resume", action="store_true")

args = parser.parse_args()

if args.task == "kernel":
    args.path = "exp/normal2kernel"
    args.dynamic_batch_size = 1
    args.n_epochs = 120
    args.base_lr = 3e-2
    args.warmup_epochs = 5
    args.warmup_lr = -1
    args.ks_list = "3,5,7"
    args.expand_list = "6"
    args.depth_list = "4"
elif args.task == "depth":
    args.path = "exp/kernel2kernel_depth/phase%d" % args.phase
    args.dynamic_batch_size = 2
    if args.phase == 1:
        args.n_epochs = 25
        args.base_lr = 2.5e-3
        args.warmup_epochs = 0
        args.warmup_lr = -1
        args.ks_list = "3,5,7"
        # args.expand_list = "6"
        args.expand_list = "1"
        args.depth_list = "3,4"
    else:
        args.n_epochs = 120
        args.base_lr = 7.5e-3
        args.warmup_epochs = 5
        args.warmup_lr = -1
        args.ks_list = "3,5,7"
        args.expand_list = "6"
        args.depth_list = "2,3,4"
elif args.task == "expand":
    args.path = "exp/kernel_depth2kernel_depth_width/phase%d" % args.phase
    args.dynamic_batch_size = 4
    if args.phase == 1:
        args.n_epochs = 25
        args.base_lr = 2.5e-3
        args.warmup_epochs = 0
        args.warmup_lr = -1
        args.ks_list = "3,5,7"
        args.expand_list = "4,6"
        args.depth_list = "2,3,4"
    else:
        args.n_epochs = 120
        args.base_lr = 7.5e-3
        args.warmup_epochs = 5
        args.warmup_lr = -1
        args.ks_list = "3,5,7"
        args.expand_list = "3,4,6"
        args.depth_list = "2,3,4"
else:
    raise NotImplementedError

args.image_size = "128,160,192,224"
args.continuous_size = True
args.not_sync_distributed_image_size = False

args.momentum = 0.9
args.no_nesterov = False

args.kd_ratio = 0.
args.width_mult_list = "1.0"
args.dy_conv_scaling_mode = 1

args.teacher_model = None
args.fp16_allreduce = False

args.validation_frequency = 1

##########################
if __name__ == "__main__":
    os.makedirs(args.path, exist_ok=True)

    # Initialize Horovod for Multi-GPU
    hvd.init()

    # set random seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    torch.cuda.set_device(hvd.local_rank())

    num_gpus = hvd.size()

    # image size
    args.image_size = [int(img_size) for img_size in args.image_size.split(",")]
    if len(args.image_size) == 1:
        args.image_size = args.image_size[0]

    '''
    RandomResizedCrip
    CONTINUOUS:
        Generate candidates for all sizes between the minimum and maximum sizes
        For example, if IMAGE_SIZE_LIST is set to [224, 256] and IMAGE_SIZE_SEG is 4, then the candidate sizes would be [224, 228, 232, ..., 256].
    SYNC_DISTRIBUTED:
        - False
            Each process uses an independent seed (based on the process ID and the current time) to sample image sizes.
            This means that each process can independently select images of different sizes.
    '''
    MyRandomResizedCrop.CONTINUOUS = args.continuous_size
    MyRandomResizedCrop.SYNC_DISTRIBUTED = not args.not_sync_distributed_image_size

    # build run config from args
    args.lr_schedule_param = None
    args.opt_param = {
        "momentum": args.momentum,
        "nesterov": not args.no_nesterov,
    }
    args.init_lr = args.base_lr * num_gpus  # linearly rescale the learning rate
    if args.warmup_lr < 0:
        args.warmup_lr = args.base_lr

    args.train_batch_size = args.batch_size
    args.test_batch_size = args.batch_size * 2

    run_config = DistributedImageNetRunConfig(
        **args.__dict__, num_replicas=num_gpus, rank=hvd.rank()
    )

    # # print run config information
    # if hvd.rank() == 0:
    #     print("Run config:")
    #     for k, v in run_config.config.items():
    #         print("\t%s: %s" % (k, v))

    '''
    DynamicSeparableConv2d
    dy_conv_scaling_mode:
        - 1:
            It allows for easy handling of various kernel sizes, enabling the network to have a receptive field of diverse sizes.
        - None:
            It performs standard convolution operations without using a complex mechanism for dynamically transforming kernels of various sizes.
    '''
    if args.dy_conv_scaling_mode == -1:
        args.dy_conv_scaling_mode = None
    DynamicSeparableConv2d.KERNEL_TRANSFORM_MODE = args.dy_conv_scaling_mode    # dy_conv_scaling_mode = 1

    # build net from args
    args.width_mult_list = [
        float(width_mult) for width_mult in args.width_mult_list.split(",")
    ]
    args.width_mult_list = (
        args.width_mult_list[0]
        if len(args.width_mult_list) == 1
        else args.width_mult_list
    )
    args.ks_list = [int(ks) for ks in args.ks_list.split(",")]
    args.expand_list = [int(e) for e in args.expand_list.split(",")]
    args.depth_list = [int(d) for d in args.depth_list.split(",")]

    # network
    net = OFAResNets(
        n_classes = args.n_classes, 
        depth_list = args.depth_list,
        expand_ratio_list = args.expand_list,
        width_mult_list = args.width_mult_list
    )

    """ Distributed RunManager """
    # Horovod: (optional) compression algorithm.
    compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none # applies a compression method using 16-bit floating-point format
    distributed_run_manager = DistributedRunManager(
        args.path,
        net,
        run_config,
        compression,
        backward_steps=args.dynamic_batch_size,
        is_root=(hvd.rank() == 0),
    )
    distributed_run_manager.save_config()
    # hvd broadcast
    distributed_run_manager.broadcast()

    # training
    from ofa.modules.progressive_shrinking import (
        validate,
        train,
    )

    validate_func_dict = {
        "image_size_list": {224}
        if isinstance(args.image_size, int)
        else sorted({160, 224}),
        "ks_list": sorted({min(args.ks_list), max(args.ks_list)}),
        "expand_ratio_list": sorted({min(args.expand_list), max(args.expand_list)}),
        "depth_list": sorted({min(net.depth_list), max(net.depth_list)}),
    }
    if args.task == "kernel":
        validate_func_dict["ks_list"] = sorted(args.ks_list)

        train(
            distributed_run_manager,
            args,
            lambda _run_manager, epoch, is_test: validate(
                _run_manager, epoch, is_test, **validate_func_dict
            ),
        )
    elif args.task == "depth":
        from ofa.modules.progressive_shrinking import (
            train_elastic_depth,
        )

        train_elastic_depth(train, distributed_run_manager, args, validate_func_dict)
    elif args.task == "expand":
        from ofa.modules.progressive_shrinking import (
            train_elastic_expand,
        )

        train_elastic_expand(train, distributed_run_manager, args, validate_func_dict)
    else:
        raise NotImplementedError
