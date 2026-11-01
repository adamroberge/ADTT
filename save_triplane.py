import time, argparse, os.path as osp, os
import torch, numpy as np
import torch.distributed as dist
from copy import deepcopy

import mmcv
from mmengine import Config
from mmengine.runner import set_random_seed
from mmengine.optim import build_optim_wrapper
from mmengine.logging import MMLogger
from mmengine.utils import symlink
from mmengine.registry import MODELS
from timm.scheduler import CosineLRScheduler, MultiStepLRScheduler
from utils.load_save_util import revise_ckpt, revise_ckpt_1
import warnings
warnings.filterwarnings("ignore")

import shutil

import sys, os, pdb

class ForkedPdb(pdb.Pdb):
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin

def pass_print(*args, **kwargs):
    pass

def main(local_rank, args):
    set_random_seed(args.seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    cfg = Config.fromfile(args.py_config)
    cfg.work_dir = args.work_dir

    if args.gpus > 1:
        distributed = True
        ip = os.environ.get("MASTER_ADDR", "127.0.0.1")
        port = os.environ.get("MASTER_PORT", cfg.get("port", 29995))
        hosts = int(os.environ.get("WORLD_SIZE", 1))
        rank = int(os.environ.get("RANK", 0))
        gpus = torch.cuda.device_count()
        print(f"tcp://{ip}:{port}")
        dist.init_process_group(
            backend="nccl", init_method=f"tcp://{ip}:{port}",
            world_size=hosts * gpus, rank=rank * gpus + local_rank)
        world_size = dist.get_world_size()
        cfg.gpu_ids = range(world_size)
        torch.cuda.set_device(local_rank)

        if local_rank != 0:
            import builtins
            builtins.print = pass_print
    else:
        distributed = False
        world_size = 1

    if local_rank == 0:
        os.makedirs(args.work_dir, exist_ok=True)
        cfg.dump(osp.join(args.work_dir, osp.basename(args.py_config)))
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(args.work_dir, f'{timestamp}.log')
    logger = MMLogger('genocc', log_file=log_file)
    MMLogger._instance_dict['genocc'] = logger
    logger.info(f'Config:\n{cfg.pretty_text}')

    import model
    from dataset import get_dataloader, get_nuScenes_label_name
    from loss import OPENOCC_LOSS
    from utils.metric_util import MeanIoU, multi_step_MeanIou
    from utils.freeze_model import freeze_model

    my_model = MODELS.build(cfg.model)
    my_model.init_weights()
    n_parameters = sum(p.numel() for p in my_model.parameters() if p.requires_grad)
    logger.info(f'Number of params: {n_parameters}')
    if cfg.get('freeze_dict', False):
        logger.info(f'Freezing model according to freeze_dict:{cfg.freeze_dict}')
        freeze_model(my_model, cfg.freeze_dict)
    n_parameters = sum(p.numel() for p in my_model.parameters() if p.requires_grad)
    logger.info(f'Number of params after freezed: {n_parameters}')
    if distributed:
        if cfg.get('syncBN', True):
            my_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(my_model)
            logger.info('converted sync bn.')

        find_unused_parameters = cfg.get('find_unused_parameters', False)
        ddp_model_module = torch.nn.parallel.DistributedDataParallel
        my_model = ddp_model_module(
            my_model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
        raw_model = my_model.module
    else:
        my_model = my_model.cuda()
        raw_model = my_model
    logger.info('done ddp model')

    train_dataset_loader, val_dataset_loader = get_dataloader(
        cfg.train_dataset_config,
        cfg.val_dataset_config,
        cfg.train_wrapper_config,
        cfg.val_wrapper_config,
        cfg.train_loader,
        cfg.val_loader,
        dist=distributed,
        iter_resume=args.iter_resume,
        train_sampler_config=cfg.train_loader,
        val_sampler_config=cfg.val_loader,
    )

    optimizer = build_optim_wrapper(my_model, cfg.optimizer)
    loss_func = OPENOCC_LOSS.build(cfg.loss).cuda()
    max_num_epochs = cfg.max_epochs
    if cfg.get('multisteplr', False):
        scheduler = MultiStepLRScheduler(
            optimizer,
            **cfg.multisteplr_config)
    else:
        scheduler = CosineLRScheduler(
            optimizer,
            t_initial=len(train_dataset_loader) * max_num_epochs,
            lr_min=1e-6,
            warmup_t=cfg.get('warmup_iters', 500),
            warmup_lr_init=1e-6,
            t_in_epochs=False)

    epoch = 0
    global_iter = 0
    last_iter = 0
    best_val_iou = [0]*cfg.get('return_len_', 10)
    best_val_miou = [0]*cfg.get('return_len_', 10)

    cfg.resume_from = ''
    if osp.exists(osp.join(args.work_dir, 'latest.pth')):
        cfg.resume_from = osp.join(args.work_dir, 'latest.pth')
    if args.resume_from:
        cfg.resume_from = args.resume_from

    logger.info('resume from: ' + cfg.resume_from)
    logger.info('work dir: ' + args.work_dir)

    if cfg.resume_from and osp.exists(cfg.resume_from):
        map_location = 'cpu'
        ckpt = torch.load(cfg.resume_from, map_location=map_location)
        print(raw_model.load_state_dict(ckpt['state_dict'], strict=False))
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        epoch = ckpt['epoch']
        global_iter = ckpt['global_iter']
        last_iter = ckpt['last_iter'] if 'last_iter' in ckpt else 0
        if 'best_val_iou' in ckpt:
            best_val_iou = ckpt['best_val_iou']
        if 'best_val_miou' in ckpt:
            best_val_miou = ckpt['best_val_miou']

        if hasattr(train_dataset_loader.sampler, 'set_last_iter'):
            train_dataset_loader.sampler.set_last_iter(last_iter)
        print(f'successfully resumed from epoch {epoch}')

    elif cfg.load_from:
        ckpt = torch.load(cfg.load_from, map_location='cpu')
        if 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt
        if cfg.get('revise_ckpt', False):
            if cfg.revise_ckpt == 1:
                print('revise_ckpt')
                print(raw_model.load_state_dict(revise_ckpt(state_dict), strict=False))
            elif cfg.revise_ckpt == 2:
                print('revise_ckpt_1')
                print(raw_model.load_state_dict(revise_ckpt_1(state_dict), strict=False))
            elif cfg.revise_ckpt == 3:
                print('revise_ckpt_2')
                print(raw_model.vae.load_state_dict(state_dict, strict=False))
        else:
            print(raw_model.load_state_dict(state_dict, strict=False))

    time.sleep(5)
    logger.info("Save Train:")

    my_model.eval()
    os.environ['eval'] = 'true'

    data_path = cfg.train_dataset_config['data_path']
    input_dataset = cfg.train_dataset_config['input_dataset']

    for i_iter, (input_occs, _, querys, xyz_labels, xyz_centers, metas) \
            in enumerate(train_dataset_loader):

        scene_name = metas[0]['scene_name']
        assert len(metas[0]['scene_token']) == metas[0]['scene_length']

        logger.info("now at {}-th {}, {} scenes at total.".format(i_iter+1, scene_name, len(train_dataset_loader)))

        for iii in range(metas[0]['scene_length']):
            one_occ = input_occs[0, iii].unsqueeze(0).unsqueeze(0).cuda()
            one_query = querys[0][iii].unsqueeze(0).cuda()
            one_xyzlabel = xyz_labels[0][iii].unsqueeze(0).cuda()
            one_xyzcenter = xyz_centers[0][iii].unsqueeze(0).cuda()

            result_dict = my_model(
                x=one_occ, querys=one_query,
                xyz_labels=one_xyzlabel, xyz_centers=one_xyzcenter, metas=metas)

            scene_token = metas[0]['scene_token'][iii]
            save_path = f'{input_dataset}/{scene_name}/{scene_token}'
            triplane_file = os.path.join(data_path, save_path, 'triplane.pth')
            torch.save(result_dict, triplane_file)
            assert len(metas[0][iii]) == 6
            for one_img_path in metas[0][iii]:
                shutil.copy(one_img_path, os.path.join(data_path, save_path))
            print("\r\t{}/{}: {}".format(iii+1, metas[0]['scene_length'], triplane_file), end="")
        print()

    logger.info("Save Eval:")
    with torch.no_grad():
        for i_iter_val, (input_occs, _, querys, xyz_labels, xyz_centers, metas) \
                in enumerate(val_dataset_loader):
            scene_name = metas[0]['scene_name']
            assert len(metas[0]['scene_token']) == metas[0]['scene_length']

            logger.info("now at {}-th {}, {} scenes at total.".format(i_iter_val+1, scene_name, len(val_dataset_loader)))

            for iii in range(metas[0]['scene_length']):

                one_occ = input_occs[0, iii].unsqueeze(0).unsqueeze(0).cuda()
                one_query = querys[0][iii].unsqueeze(0).cuda()
                one_xyzlabel = xyz_labels[0][iii].unsqueeze(0).cuda()
                one_xyzcenter = xyz_centers[0][iii].unsqueeze(0).cuda()

                result_dict = my_model(
                    x=one_occ, querys=one_query,
                    xyz_labels=one_xyzlabel, xyz_centers=one_xyzcenter, metas=metas)

                scene_token = metas[0]['scene_token'][iii]
                save_path = f'{input_dataset}/{scene_name}/{scene_token}'

                triplane_file = os.path.join(data_path, save_path, 'triplane.pth')
                torch.save(result_dict, triplane_file)

                assert len(metas[0][iii]) == 6
                for one_img_path in metas[0][iii]:
                    shutil.copy(one_img_path, os.path.join(data_path, save_path))
                print("\r\t{}/{}: {}".format(iii+1, metas[0]['scene_length'], triplane_file), end="")
            print()

    torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--py-config', default='config/tpv_lidarseg.py')
    parser.add_argument('--work-dir', type=str, default='./out/tpv_lidarseg')
    parser.add_argument('--resume-from', type=str, default='')
    parser.add_argument('--iter-resume', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    ngpus = torch.cuda.device_count()
    args.gpus = ngpus
    print(args)

    if ngpus > 1:
        torch.multiprocessing.spawn(main, args=(args,), nprocs=args.gpus)
    else:
        main(0, args)
