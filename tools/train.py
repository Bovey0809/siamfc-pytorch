from __future__ import absolute_import

import os
import sys
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from got10k.datasets import *
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from siamfc import TrackerSiamFC
from siamfc.transforms import SiamFCTransforms
from siamfc.datasets import Pair

def main():
    # 初始化分布式环境
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    
    # 创建模型
    tracker = TrackerSiamFC()
    tracker.net = tracker.net.to(local_rank)
    tracker.net = DDP(tracker.net, device_ids=[local_rank])

    # 设置数据集
    root_dir = '/home/houbowei/full_data/'
    seqs = GOT10k(root_dir, subset='train', return_meta=True)

    # 设置数据转换
    transforms = SiamFCTransforms(
        exemplar_sz=tracker.cfg.exemplar_sz,
        instance_sz=tracker.cfg.instance_sz,
        context=tracker.cfg.context)

    # 创建数据集
    dataset = Pair(
        seqs=seqs,
        transforms=transforms)

    # 创建分布式采样器
    sampler = DistributedSampler(dataset)

    # 创建数据加载器
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=tracker.cfg.batch_size,
        sampler=sampler,
        num_workers=tracker.cfg.num_workers,
        pin_memory=True,
        drop_last=True)
    
    # 开始训练
    tracker.train(seqs=seqs, 
                 save_dir='pretrained',
                 local_rank=local_rank)

if __name__ == '__main__':
    main()
