#!/bin/bash

use_ood="f"

job="dynamics_traj_cifar10"
srun -p RTX3090 -K -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-gpu=4 --mem=64G \
                --kill-on-bad-exit --job-name ${job} --nice=0 --time 5-00:00:00 \
                --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.06-py3.sqsh \
                --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
                /opt/conda/bin/python metadata_archaeology.py cifar10 ${use_ood} > ./logs/${job}.log 2>&1 &

job="dynamics_traj_cifar100"
srun -p RTX3090 -K -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-gpu=4 --mem=64G \
                --kill-on-bad-exit --job-name ${job} --nice=0 --time 5-00:00:00 \
                --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.06-py3.sqsh \
                --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
                /opt/conda/bin/python metadata_archaeology.py cifar100 ${use_ood} > ./logs/${job}.log 2>&1 &

job="dynamics_traj_imagenet"
srun -p batch -K -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-gpu=4 --mem=64G \
                --kill-on-bad-exit --job-name ${job} --nice=0 --time 5-00:00:00 \
                --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.06-py3.sqsh \
                --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
                /opt/conda/bin/python metadata_archaeology.py imagenet ${use_ood} > ./logs/${job}.log 2>&1 &
