import os
import argparse
import importlib.util
import torch.multiprocessing as mp
import torch
from isegm.utils.exp import init_experiment
import torch.distributed as dist
import time
import warnings
warnings.filterwarnings("ignore")

def main(rank, world_size, cfg, args):
    # 让线程其他线程先休眠1秒,确保线程0先启动
    if rank != 0:
        time.sleep(1)
    dist.init_process_group(backend='nccl', world_size=world_size, rank=rank)
    torch.cuda.set_device(rank)  # 解决卡0多出很多 731MB 显存占用的情况
    model_script = load_module(args.model_path)
    model_script.main(cfg, rank)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('model_path', default='models/gnnvit/plainvit_base448_cocolvis_itermask.py', type=str,
                        help='Path to the model script.')

    parser.add_argument('--model_base_name', default='SimpleClick-B_cloud', type=str,)

    parser.add_argument('--exp-name', type=str, default='',
                        help='Here you can specify the name of the experiment. '
                             'It will be added as a suffix to the experiment folder.')

    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='Dataloader threads.')

    parser.add_argument('--batch-size', type=int, default=-1,
                        help='You can override model batch size by specify positive number.')

    parser.add_argument('--ngpus', type=int, default=1,
                        help='Number of GPUs. '
                             'If you only specify "--gpus" argument, the ngpus value will be calculated automatically. '
                             'You should use either this argument or "--gpus".')

    parser.add_argument('--gpus', type=str, default='2', required=False,
                        help='Ids of used GPUs. You should use either this argument or "--ngpus".')

    parser.add_argument('--resume-exp', type=str, default=None,
                        help='The prefix of the name of the experiment to be continued. '
                             'If you use this field, you must specify the "--resume-prefix" argument.')

    parser.add_argument('--resume-prefix', type=str, default='latest',
                        help='The prefix of the name of the checkpoint to be loaded.')

    parser.add_argument('--start-epoch', type=int, default=0,
                        help='The number of the starting epoch from which training will continue. '
                             '(it is important for correct logging and learning rate)')

    parser.add_argument('--weights', type=str, default=None,
                        help='Model weights will be loaded from the specified path if you use this argument.')

    parser.add_argument('--temp-model-path', type=str, default='',
                        help='Do not use this argument (for internal purposes).')

    parser.add_argument("--local_rank", type=int, default=0)

    # parameters for experimenting
    parser.add_argument('--layerwise-decay', action='store_true',
                        help='layer wise decay for transformer blocks.')

    parser.add_argument('--upsample', type=str, default='x1',
                        help='upsample the output.')

    parser.add_argument('--random-split', action='store_true',
                        help='random split the patch instead of window split.')

    return parser.parse_args()


def load_module(script_path):
    spec = importlib.util.spec_from_file_location("model_script", script_path)
    model_script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_script)

    return model_script


if __name__ == '__main__':
    devices = '0'

    args = parse_args()
    args.distributed = True
    args.gpus = devices

    cfg = init_experiment(args, args.model_base_name)

    world_size = cfg.ngpus
    os.environ['CUDA_VISIBLE_DEVICES'] = devices
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29513'
    torch.backends.cudnn.benchmark = True
    torch.multiprocessing.set_sharing_strategy('file_system')

    # os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'      # 可以让报错信息更加准确
    # os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

    mp.spawn(main,
        args=(world_size,cfg, args),
        nprocs=world_size,
        join=True)