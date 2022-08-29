import pickle
from collections import OrderedDict

import torch
import apex.parallel


def is_main_proc(local_rank=None, shared_fs=True):
    assert shared_fs or local_rank is not None
    main_proc = not torch.distributed.is_initialized() or (torch.distributed.get_rank() == 0 if shared_fs else local_rank == 0)
    return main_proc


def get_world_size():
    return torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1


def dist_print(*msg, main_proc=None):
    if main_proc is None:
        main_proc = is_main_proc()
    if main_proc:
        print(*msg)


def wait_for_other_procs():
    if torch.distributed.is_initialized():
        torch.distributed.barrier()


def broadcast_from_main(tensor, is_tensor=True):
    if not torch.distributed.is_initialized() or torch.distributed.get_world_size() == 1:
        return tensor
    
    if is_tensor:
        tensor = tensor.cuda()
    else:
        # Serialize data to a Tensor
        buffer = pickle.dumps(tensor)
        storage = torch.ByteStorage.from_buffer(buffer)
        tensor = torch.ByteTensor(storage).cuda()
    
    torch.distributed.broadcast(tensor, src=0)
    assert (reduce_tensor(tensor, average=True) - tensor <= 1e-6).all()
    return tensor


def reduce_tensor(tensor, average=False):
    world_size = get_world_size()
    if world_size == 1:
        return tensor
    rt = tensor.clone()
    torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
    if average:
        rt /= world_size
    return rt


# def gather_tensor(tensor):
#     if not torch.distributed.is_initialized() or torch.distributed.get_world_size() == 1:
#         return tensor
#     tensor_device = tensor.device
#     tensor = tensor.cuda()
#     tensor_list = [torch.zeros_like(tensor).cuda() for _ in range(torch.distributed.get_world_size())]
#     torch.distributed.all_gather(tensor_list, tensor)
#     tensor = torch.cat(tensor_list, dim=0).to(tensor_device)
#     return tensor


def gather_tensor(data):
    """
    Imported from: https://github.com/facebookresearch/detr/blob/a54b77800eb8e64e3ad0d8237789fcbf2f8350c5/util/misc.py#L88
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    if torch.is_tensor(data):
        main_device = data.device
    else:
        main_device = torch.device("cuda")
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.tensor([tensor.numel()], device="cuda")
    size_list = [torch.tensor([0], device="cuda") for _ in range(world_size)]
    torch.distributed.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device="cuda"))
    if local_size != max_size:
        padding = torch.empty(size=(max_size - local_size,), dtype=torch.uint8, device="cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    torch.distributed.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer).to(main_device))
    return data_list


def convert_state_dict(state_dict, require_module=None):
    # Create new OrderedDict from the checkpoint state that does or does not contain "module." based on the model state
    if require_module is None:
        require_module = torch.distributed.is_initialized()
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if require_module and not name.startswith("module."):
            name = "module." + k  # add module.
        elif not require_module and name.startswith("module."):
            name = k[7:]  # remove module.
        new_state_dict[name] = v
    return new_state_dict


def convert_to_distributed(model, local_rank, sync_bn=False, use_torch_ddp=True):
    # Convert the model to dist
    dist_print(f"Using {'Torch' if use_torch_ddp else 'APEX'} DDP...")
    if torch.distributed.is_initialized():
        if sync_bn:
            dist_print("Using synced BN!")
            if use_torch_ddp:
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            else:
                model = apex.parallel.convert_syncbn_model(model)

        dist_print("Wrapping the model into DDP!")
        if use_torch_ddp:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
        else:
            model = apex.parallel.DistributedDataParallel(model, delay_allreduce=True)
    return model
