import argparse
import json
import copy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets.shapenet import build_shapenet
from models.nerf import build_nerf
from models.rendering import get_rays_shapenet, sample_points, volume_render
import wandb
from shapenet_test import test
from rich import print
from rich import pretty
pretty.install()
from rich import traceback
traceback.install()
from torchvision.utils import make_grid
from utils.shape_video import create_360_video
from pathlib import Path
import numpy as np
import random
SEED=42
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
import logging

def inner_loop(args, model, optim, imgs, poses, hwf, bound, num_samples, raybatch_size, inner_steps,
               device, idx, log_round=False, setup="train/"):
    """
    train the inner model for a specified number of iterations
    """
    pixels = imgs.reshape(-1, 3)

    rays_o, rays_d = get_rays_shapenet(hwf, poses)
    rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)

    num_rays = rays_d.shape[0]
    logs = dict()
    for i in range(1, inner_steps+1):
        if log_round and ((i % args.tto_log_steps == 0) or (i == inner_steps) or (i==1)):
            with torch.no_grad():
                scene_psnr = report_result(model, imgs,
                                           poses, hwf,
                                           bound, num_samples, raybatch_size)

                vid_frames = create_360_video(args, model, hwf, bound,
                                              device,
                                               idx, args.savedir)
                logs[setup + "scene_psnr tto_step=" + str(i)] = scene_psnr
                logs[setup + "vid_post tto_step=" + str(i)] = wandb.Video(
                            vid_frames.transpose(0, 3, 1, 2), fps=30,
                            format="mp4")

        indices = torch.randint(num_rays, size=[raybatch_size])
        raybatch_o, raybatch_d = rays_o[indices], rays_d[indices]
        pixelbatch = pixels[indices] 
        t_vals, xyz = sample_points(raybatch_o, raybatch_d, bound[0], bound[1],
                                    num_samples, perturb=True)
        
        optim.zero_grad()
        rgbs, sigmas = model(xyz)
        colors = volume_render(rgbs, sigmas, t_vals, white_bkgd=True)
        loss = F.mse_loss(colors, pixelbatch)
        loss.backward()
        optim.step()
    return logs


def train_meta(args, epoch_idx, meta_model, meta_optim, data_loader, device):
    """
    train the meta_model for one epoch using reptile meta learning
    https://arxiv.org/abs/1803.02999
    """

    step = (epoch_idx - 1) * len(data_loader)
    avg_psnr = 0
    psnr_accum = dict()
    for idx,(imgs, poses, hwf, bound) in enumerate(data_loader):
        log_round = (step % args.log_interval == 0)
        imgs, poses, hwf, bound = imgs.to(device), poses.to(device), hwf.to(device), bound.to(device)
        imgs, poses, hwf, bound = imgs.squeeze(), poses.squeeze(), hwf.squeeze(), bound.squeeze()

        meta_optim.zero_grad()

        inner_model = copy.deepcopy(meta_model)
        inner_optim = torch.optim.SGD(inner_model.parameters(), args.inner_lr)

        logs=inner_loop(args, inner_model, inner_optim, imgs, poses,
                    hwf, bound, args.num_samples,
                    args.train_batchsize, args.inner_steps, device=device, idx=idx, log_round=log_round,
                   setup="train/")
        
        with torch.no_grad():
            for meta_param, inner_param in zip(meta_model.parameters(), inner_model.parameters()):
                meta_param.grad = meta_param - inner_param
        meta_optim.step()

        if log_round:
            avg_psnr += logs["train/scene_psnr tto_step=" + str(args.inner_steps)]
            # logs["train/gen_model_mse_loss"] = float(loss)
            logs = {**logs,  "train_step": step,
                    "train/imgs": wandb.Image(
                        make_grid(imgs.permute(0, 3, 1, 2)))}
            wandb.log(logs)
            for (key, val) in logs.items():
                if "psnr" in key:
                    if psnr_accum.get(key) is None:
                        psnr_accum[key] = 0
                    psnr_accum[key] += val
        step+=1
    psnr_mean = dict()
    for (key, val) in psnr_accum.items():
        psnr_mean[key + "_mean"] = val / len(data_loader)
    avg_psnr /= len(data_loader)
    wandb.log({**psnr_mean, "val/avg_psnr": avg_psnr, "epoch_step": epoch_idx})


def report_result(model, imgs, poses, hwf, bound, num_samples, raybatch_size):
    """
    report view-synthesis result on heldout views
    """
    ray_origins, ray_directions = get_rays_shapenet(hwf, poses)

    view_psnrs = []
    for img, rays_o, rays_d in zip(imgs, ray_origins, ray_directions):
        rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)
        t_vals, xyz = sample_points(rays_o, rays_d, bound[0], bound[1],
                                    num_samples, perturb=False)
        
        synth = []
        num_rays = rays_d.shape[0]
        with torch.no_grad():
            for i in range(0, num_rays, raybatch_size):
                rgbs_batch, sigmas_batch = model(xyz[i:i+raybatch_size])
                color_batch = volume_render(rgbs_batch, sigmas_batch, 
                                            t_vals[i:i+raybatch_size],
                                            white_bkgd=True)
                synth.append(color_batch)
            synth = torch.cat(synth, dim=0).reshape_as(img)
            error = F.mse_loss(img, synth)
            psnr = -10*torch.log10(error)
            view_psnrs.append(psnr)
    
    scene_psnr = torch.stack(view_psnrs).mean()
    return scene_psnr


def val_meta(args, epoch_idx, model, val_loader, device):
    """
    validate the meta trained model for few-shot view synthesis
    """
    meta_trained_state = model.state_dict()
    val_model = copy.deepcopy(model)
    avg_psnr = 0
    psnr_accum = dict()
    val_step = max((epoch_idx - 1) * len(val_loader) + 1, 0)
    for idx, (imgs, poses, hwf, bound) in enumerate(val_loader):
        imgs, poses, hwf, bound = imgs.to(device), poses.to(device), hwf.to(device), bound.to(device)
        imgs, poses, hwf, bound = imgs.squeeze(), poses.squeeze(), hwf.squeeze(), bound.squeeze()

        tto_imgs, test_imgs = torch.split(imgs, [args.tto_views, args.test_views], dim=0)
        tto_poses, test_poses = torch.split(poses, [args.tto_views, args.test_views], dim=0)

        val_model.load_state_dict(meta_trained_state)
        val_optim = torch.optim.SGD(val_model.parameters(), args.tto_lr)

        logs = inner_loop(args, val_model, val_optim, tto_imgs, tto_poses, hwf,
                    bound, args.num_samples, args.tto_batchsize, args.tto_steps,
                    device=device, idx=idx, log_round=True, setup="val/")

        avg_psnr += logs["val/scene_psnr tto_step=" + str(args.tto_steps)]
        logs["val/tto_views"] = wandb.Image(
            make_grid(tto_imgs.permute(0, 3, 1, 2)))
        logs["val/test_views"] = wandb.Image(
            make_grid(test_imgs.permute(0, 3, 1, 2)))
        logs["val_step"] = val_step
        wandb.log(logs)
        for (key,val) in logs.items():
            if "psnr" in key:
                if psnr_accum.get(key) is None:
                    psnr_accum[key] = 0
                psnr_accum[key] += val
        val_step+=1

    psnr_mean = dict()
    for (key,val) in psnr_accum.items():
        psnr_mean[key+"_mean"] = val/len(val_loader)
    avg_psnr /= len(val_loader)
    wandb.log({**psnr_mean, "val/avg_psnr":avg_psnr, "epoch_step":epoch_idx})

def main():
    parser = argparse.ArgumentParser(description='shapenet few-shot view synthesis')
    parser.add_argument('--config', type=str, required=True,
                        help='config file for the shape class (cars, chairs or lamps)')
    parser.add_argument('--weight_path', type=str, default=None)
    args = parser.parse_args()

    with open(args.config) as config:
        info = json.load(config)
        for key, value in info.items():
            args.__dict__[key] = value
    args.savedir = Path(args.savedir)
    wandb.init(name="train_"+args.exp_name, dir="/root/nerf-meta/", project="meta_NeRF", entity="stereo",
               save_code=True, job_type="train")

    wandb.config.update(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_set = build_shapenet(image_set="train", dataset_root=args.dataset_root,
                                splits_path=args.splits_path, num_views=args.train_views)
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True)

    val_set = build_shapenet(image_set="val", dataset_root=args.dataset_root,
                            splits_path=args.splits_path,
                            num_views=args.tto_views+args.test_views)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

    meta_model = build_nerf(args)
    meta_model.to(device)

    if hasattr(args, "weight_path") and args.weight_path is not None:
        checkpoint = torch.load(args.weight_path, map_location=device)
        meta_state = checkpoint['meta_model_state_dict']
        meta_model.load_state_dict(meta_state)

    meta_optim = torch.optim.Adam(meta_model.parameters(), lr=args.meta_lr)
    logging.info("starting to train...")

    val_meta(args, 0, meta_model, val_loader, device)
    for epoch in range(1, args.meta_epochs+1):
        logging.info("Epoch: " + str(epoch))
        train_meta(args, epoch, meta_model, meta_optim, train_loader, device)
        val_meta(args, epoch, meta_model, val_loader, device)

        ckpt_name = args.save_dir + "/"+args.exp_name+"_epoch" + str(epoch) + ".pth"
        torch.save({
            'epoch': epoch,
            'meta_model_state_dict': meta_model.state_dict(),
            'meta_optim_state_dict': meta_optim.state_dict(),
            }, ckpt_name)
        wandb.save(ckpt_name)
        args.weight_path = ckpt_name
    test(args)

if __name__ == '__main__':
    main()