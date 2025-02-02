import argparse
import json
import copy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.weight_generator import WeightGenerator, add_weight_res
from models.nerf import build_nerf, set_grad
from models.rendering import get_rays_shapenet, sample_points, volume_render
from datasets.shapenetV2 import build_shapenetV2
import wandb
from weight_gen_test import test
from rich import print
from rich import pretty
pretty.install()
from rich import traceback
traceback.install()
from utils.shape_video import create_360_video
from pathlib import Path
from torchvision.utils import make_grid
import torch.nn as nn
import numpy as np
import random
SEED=42
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
import lpips
import pytorch_ssim
import logging
import hydra
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig
import os

def make_img_idx(available_views, num_views, num_runs):
    idx_list = []
    finished = 0
    idxs = list(range(available_views))
    while finished < num_runs:
        i = np.random.choice(idxs, size=num_views)
        i.sort()
        # if i not in idx_list:
        if len(np.where((idx_list == i))[0]) == 0:
            idx_list.append(i)
            finished +=1
    return idx_list


def split_at_idx(t, idx=[0]):


    t1 = []
    t2 = []
    b = t.size()[0]
    for i in range(b):
        if i in idx:
            t1.append(t[i])
        else:
            t2.append(t[i])

    t1 = torch.stack(t1, dim=0)
    t2 = torch.stack(t2, dim=0)
    return t1, t2

def inner_loop(args, nerf_model, nerf_optim, pixels, imgs, rays_o, rays_d,
               poses, bound, hwf, num_samples, raybatch_size, inner_steps,
               device, idx, log_round=False, setup="train/", input_idx=[0]):
    """
    train the inner model for a specified number of iterations
    """
    num_rays = rays_d.shape[0]
    logs = dict()
    for i in range(1, inner_steps+1):
        if log_round and ((i % args.nerf.tto_log_steps == 0) or (i == inner_steps) or (i==1)):
            with torch.no_grad():
                scene_psnr, scene_lpips_alex, scene_lpips_vgg, scene_ssims =\
                    0, 0, 0, 0
                scene_psnr, scene_lpips_alex, scene_lpips_vgg, scene_ssims\
                            = report_result(nerf_model, imgs,
                                           poses, hwf,
                                           bound, num_samples, raybatch_size)
                vid_save_path = os.path.join(cwd, "video")
                vid_frames = create_360_video(args.nerf, nerf_model, hwf, bound,
                                              device,
                                               idx, vid_save_path)

                if "train" in setup:
                    logs[setup + "SSIM tto_step=" + str(i)] = scene_ssims
                    logs[setup + "LPIPS_vgg tto_step=" + str(i)] = scene_lpips_vgg
                    logs[setup + "LPIPS_alexnet tto_step=" + str(i)] = scene_lpips_alex
                    logs[setup + "scene_psnr tto_step=" + str(i)] = scene_psnr
                    logs[setup + "vid_post tto_step=" + str(i)] = wandb.Video(
                                vid_frames.transpose(0, 3, 1, 2), fps=30,
                                format="mp4")
                else:
                    logs[setup + "SSIM tto_step=" + str(i)] = scene_ssims
                    logs[setup + "LPIPS_vgg tto_step=" + str(i)] = scene_lpips_vgg
                    logs[setup + "LPIPS_alexnet tto_step=" + str(i)] = scene_lpips_alex
                    logs[setup + "scene_psnr tto_step=" + str(i)] = scene_psnr
                    logs[setup + "vid_post input_idx=" + str(input_idx) + " tto_step=" + str(i)] = wandb.Video(
                                vid_frames.transpose(0, 3, 1, 2), fps=30,
                                format="mp4")

        indices = torch.randint(num_rays, size=[raybatch_size])
        raybatch_o, raybatch_d = rays_o[indices], rays_d[indices]
        pixelbatch = pixels[indices] 
        t_vals, xyz = sample_points(raybatch_o, raybatch_d, bound[0], bound[1],
                                    num_samples, perturb=True)
        

        rgbs, sigmas = nerf_model(xyz)
        colors = volume_render(rgbs, sigmas, t_vals, white_bkgd=True)
        loss = F.mse_loss(colors, pixelbatch)
        loss.backward()
        nerf_optim.step()
        nerf_optim.zero_grad()

    return logs, nerf_model



def train_meta(args, epoch_idx, nerf_model, gen_model, gen_optim, data_loader, device, ref_state_dict=None):
    """
    train the meta_model for one epoch using reptile meta learning
    https://arxiv.org/abs/1803.02999
    """
    gen_model.train()
    gen_model.requires_grad_(True)
    gen_model.feature_extractor.requires_grad_(False)


    step = (epoch_idx-1)*len(data_loader)
    train_step = (epoch_idx - 1) * len(data_loader) + 1
    psnr_accum = dict()
    ssim_accum = dict()
    lpips_alex_accum = dict()
    lpips_vgg_accum = dict()

    avg_psnr = 0
    avg_ssim = 0
    avg_lpips_alex = 0
    avg_lpips_vgg = 0
    for idx, batch in enumerate(data_loader):
        log_round=(step % args.log_interval == 0)

        imgs = batch["imgs"]
        poses = batch["poses"]
        hwf = batch["hwf"]
        bound = batch["bound"]
        relative_poses = batch["relative_poses"]

        imgs, poses, hwf, bound, relative_poses = imgs.to(device), poses.to(device), hwf.to(device), bound.to(device), relative_poses.to(device)
        imgs, poses, hwf, bound, relative_poses = imgs.squeeze(), poses.squeeze(), hwf.squeeze(), bound.squeeze(), relative_poses.squeeze()
        rays_o, rays_d = get_rays_shapenet(hwf, poses)
        rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)
        num_rays = rays_d.shape[0]
        pixels = imgs.reshape(-1, 3)

        # Train weight generator
        nerf_model_copy = copy.deepcopy(nerf_model) #! copy meta model initialized weights
        weight_res = gen_model(imgs, relative_poses, bound)

        nerf_model_copy, logs_weight_stat = \
            add_weight_res(nerf_model_copy, weight_res, log_round=log_round,
                           setup="train/", std_scale=args.feat.std_scale, hidden_layers=args.nerf.nerf_hidden_layers)
        indices = torch.randint(num_rays, size=[args.nerf.train_batchsize])
        raybatch_o, raybatch_d = rays_o[indices], rays_d[indices]
        pixelbatch = pixels[indices]
        t_vals, xyz = sample_points(raybatch_o, raybatch_d, bound[0], bound[1],
                                    args.nerf.num_samples, perturb=True)

        rgbs, sigmas = nerf_model_copy(xyz)
        colors = volume_render(rgbs, sigmas, t_vals, white_bkgd=True)

        loss = F.mse_loss(colors, pixelbatch)
        # loss.backward()
        # gen_optim.step()
        # gen_optim.zero_grad()

        #! this block causes problem when trying to do reptile loss since
        #! weight assignment is an in-place operation
        inner_nerf_model_copy = copy.deepcopy(nerf_model)
        for i in range(len(nerf_model_copy.net)):
            if hasattr(nerf_model_copy.net[i], "weight"):
                layer = nerf_model_copy.net[i].weight.clone()
                layer.grad=None
                inner_nerf_model_copy.net[i].weight = nn.Parameter(layer)

        inner_nerf_model_copy = set_grad(inner_nerf_model_copy, True)
        inner_nerf_model_copy.train()
        nerf_optim = torch.optim.SGD(inner_nerf_model_copy.parameters(), args.nerf.inner_lr)

        logs, tto_nerf_model = inner_loop(args, inner_nerf_model_copy, nerf_optim, pixels, imgs,
                    rays_o, rays_d, poses, bound, hwf, args.nerf.num_samples,
                    args.nerf.train_batchsize, args.nerf.inner_steps,
                    device=device, idx=idx, log_round=log_round, setup="train/")

        if args.feat.use_reptile_loss:
            gt_res = []
            for i in range(args.nerf_hidden_layers):
                l = i * 2 + 1
                gt_res.append(torch.flatten(tto_nerf_model.net[l].weight.data - nerf_model.net[l].weight.data))
            gt_res = torch.cat(gt_res).detach().requires_grad_(True)
            gt_res.grad=None
            reptile_loss= F.mse_loss(gt_res, weight_res)*args.feat.reptile_loss_weight
            loss += reptile_loss

        loss.backward()
        gen_optim.step()
        gen_optim.zero_grad()

        if log_round:
            if args.feat.use_reptile_loss:
                logs["reptile_loss"] = reptile_loss
            avg_psnr += logs["train/scene_psnr tto_step=" + str(args.nerf.inner_steps)]
            avg_lpips_alex += logs["train/LPIPS_alexnet tto_step="+ str(args.nerf.inner_steps)]
            avg_lpips_vgg += logs["train/LPIPS_vgg tto_step="+ str(args.nerf.inner_steps)]
            avg_ssim += logs["train/SSIM tto_step="+ str(args.nerf.inner_steps)]

            logs["train/gen_model_mse_loss"] = float(loss)
            logs = {**logs, **logs_weight_stat, "train_step": train_step,
                    "train/imgs":wandb.Image(make_grid(imgs.permute(0, 3, 1, 2)))}
            wandb.log(logs)

            for (key, val) in logs.items():
                if "psnr" in key:
                    if psnr_accum.get(key) is None:
                        psnr_accum[key] = 0
                    psnr_accum[key] += val

                if "LPIPS_vgg" in key:
                    if lpips_vgg_accum.get(key) is None:
                        lpips_vgg_accum[key] = 0
                    lpips_vgg_accum[key] += val

                if "LPIPS_alexnet" in key:
                    if lpips_alex_accum.get(key) is None:
                        lpips_alex_accum[key] = 0
                    lpips_alex_accum[key] += val

                if "SSIM" in key:
                    if ssim_accum.get(key) is None:
                        ssim_accum[key] = 0
                    ssim_accum[key] += val
        step+=1
    psnr_mean = dict()
    ssim_mean = dict()
    lpips_alex_mean = dict()
    lpips_vgg_mean = dict()

    for (key, val) in psnr_accum.items():
        psnr_mean[key + "_mean"] = val / len(data_loader)
    for (key, val) in lpips_alex_accum.items():
        lpips_alex_mean[key + "_mean"] = val / len(data_loader)
    for (key, val) in lpips_vgg_accum.items():
        lpips_vgg_mean[key + "_mean"] = val / len(data_loader)
    for (key, val) in ssim_accum.items():
        ssim_mean[key + "_mean"] = val / len(data_loader)

    avg_psnr /= len(data_loader)
    avg_lpips_alex /= len(data_loader)
    avg_lpips_vgg /= len(data_loader)
    avg_ssim /= len(data_loader)

    wandb.log({**psnr_mean, **lpips_alex_mean, **lpips_vgg_mean,
               **ssim_mean, "train/PSNR_epoch_mean": avg_psnr,
               "train/LPIPS_alexnet_epoch_mean":avg_lpips_alex,
               "train/LPIPS_vgg_epoch_mean":avg_lpips_vgg,
               "train/SSIM_epoch_mean":avg_ssim})

def report_result(model, imgs, poses, hwf, bound, num_samples, raybatch_size):
    """
    report view-synthesis result on heldout views
    """
    ray_origins, ray_directions = get_rays_shapenet(hwf, poses)

    view_psnrs = []
    view_lpips_alex = []
    view_lpips_vgg = []
    view_ssims = []
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

            # additional metrics (lpips alexnet, lpips vgg, ssim)
            img_lpips = torch.unsqueeze(img.permute(2, 0, 1) * 2 - 1, 0)
            synth_lpips = torch.unsqueeze(synth.permute(2, 0, 1) * 2 - 1, 0)
            view_lpips_alex.append(lpips_alex(img_lpips, synth_lpips))
            view_lpips_vgg.append(lpips_vgg(img_lpips, synth_lpips))
            view_ssims.append(pytorch_ssim.ssim(torch.unsqueeze(img, dim=0),
                                                torch.unsqueeze(synth, dim=0)))

    scene_psnr = torch.stack(view_psnrs).mean()
    scene_lpips_alex = torch.stack(view_lpips_alex).mean()
    scene_lpips_vgg = torch.stack(view_lpips_vgg).mean()
    scene_ssim = torch.stack(view_ssims).mean()
    return scene_psnr, scene_lpips_alex, scene_lpips_vgg, scene_ssim


def val_meta(args, epoch_idx, nerf_model, gen_model, val_loader, device):
    """
    validate the meta trained model for few-shot view synthesis
    """
    gen_model.eval()
    gen_model.requires_grad_(False)
    meta_trained_state = nerf_model.state_dict()

    val_step = max((epoch_idx-1)*len(val_loader) +1, 0)
    psnr_accum = dict()
    ssim_accum = dict()
    lpips_alex_accum = dict()
    lpips_vgg_accum = dict()

    avg_psnr = 0
    avg_ssim = 0
    avg_lpips_alex = 0
    avg_lpips_vgg = 0
    for idx, batch in enumerate(val_loader):
        imgs = batch["imgs"]
        poses = batch["poses"]
        hwf = batch["hwf"]
        bound = batch["bound"]
        relative_poses = batch["relative_poses"]

        imgs, poses, hwf, bound, relative_poses = imgs.to(device), poses.to(device), \
                                                  hwf.to(device), bound.to(device), relative_poses.to(device)
        imgs, poses, hwf, bound, relative_poses = imgs.squeeze(), \
                                                  poses.squeeze(), \
                                                  hwf.squeeze(), \
                                                  bound.squeeze(), relative_poses.squeeze()

        for i in range(args.val_per_scene):
            img_idx = val_views[i]
            tto_imgs, test_imgs = split_at_idx(imgs,  idx=img_idx)
            tto_poses, test_poses = split_at_idx(poses, idx=img_idx)

            rays_o, rays_d = get_rays_shapenet(hwf, tto_poses)
            rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)
            num_rays = rays_d.shape[0]

            tto_pixels = tto_imgs.reshape(-1, 3)
            # Add weight residual
            val_model = copy.deepcopy(nerf_model)
            val_model.load_state_dict(meta_trained_state)
            val_model = set_grad(val_model, False)

            with torch.no_grad():
                weight_res = gen_model(imgs[:25], relative_poses[:25], bound[:25])
                val_model, logs_weight_stat = \
                    add_weight_res(val_model, weight_res, log_round=True,
                                   setup="val/", std_scale=args.feat.std_scale)
                indices = torch.randint(num_rays, size=[args.nerf.train_batchsize])
                raybatch_o, raybatch_d = rays_o[indices], rays_d[indices]
                pixelbatch = tto_pixels[indices]
                t_vals, xyz = sample_points(raybatch_o, raybatch_d, bound[0], bound[1],
                                            args.nerf.num_samples, perturb=True)
                rgbs, sigmas = val_model(xyz)
                colors = volume_render(rgbs, sigmas, t_vals, white_bkgd=True)
                val_loss = F.mse_loss(colors, pixelbatch)


            inner_val_model = copy.deepcopy(val_model)
            for j in range(len(val_model.net)):
                if hasattr(val_model.net[j], "weight"):
                    layer = val_model.net[j].weight.clone()
                    layer.grad = None
                    inner_val_model.net[i].weight = nn.Parameter(layer)

            inner_val_model = set_grad(inner_val_model, True)
            inner_val_model.train()
            val_optim = torch.optim.SGD(inner_val_model.parameters(), args.nerf.tto_lr)


            logs,_ = inner_loop(args, inner_val_model, val_optim, tto_pixels, tto_imgs, rays_o,
                        rays_d, tto_poses, bound, hwf, args.nerf.num_samples,
                       args.nerf.tto_batchsize, args.nerf.tto_steps,
                       device=device, idx=idx, log_round=True, setup="val/", input_idx=img_idx)

            avg_psnr += logs["val/scene_psnr tto_step=" + str(args.nerf.tto_steps)]
            avg_lpips_alex += logs[
                "val/LPIPS_alexnet tto_step=" + str(args.nerf.tto_steps)]
            avg_lpips_vgg += logs[
                "val/LPIPS_vgg tto_step=" + str(args.nerf.tto_steps)]
            avg_ssim += logs["val/SSIM tto_step=" + str(args.nerf.tto_steps)]

            logs["val/tto_views"] = wandb.Image(make_grid(tto_imgs.permute(0, 3, 1, 2)))
            logs["val/test_views"] = wandb.Image(make_grid(test_imgs.permute(0, 3, 1, 2)))
            logs["val/mse_loss"] = val_loss
            logs = {**logs, **logs_weight_stat, "val_step":val_step}
            wandb.log(logs)
            for (key,val) in logs.items():
                if "psnr" in key:
                    if psnr_accum.get(key) is None:
                        psnr_accum[key] = 0
                    psnr_accum[key] += val
                if "LPIPS_vgg" in key:
                    if lpips_vgg_accum.get(key) is None:
                        lpips_vgg_accum[key] = 0
                    lpips_vgg_accum[key] += val

                if "LPIPS_alexnet" in key:
                    if lpips_alex_accum.get(key) is None:
                        lpips_alex_accum[key] = 0
                    lpips_alex_accum[key] += val

                if "SSIM" in key:
                    if ssim_accum.get(key) is None:
                        ssim_accum[key] = 0
                    ssim_accum[key] += val
        val_step+=1

    psnr_mean = dict()
    ssim_mean = dict()
    lpips_alex_mean = dict()
    lpips_vgg_mean = dict()

    for (key, val) in psnr_accum.items():
        psnr_mean[key + "_mean"] = val / len(val_loader)
    for (key, val) in lpips_alex_accum.items():
        lpips_alex_mean[key + "_mean"] = val / len(val_loader)
    for (key, val) in lpips_vgg_accum.items():
        lpips_vgg_mean[key + "_mean"] = val / len(val_loader)
    for (key, val) in ssim_accum.items():
        ssim_mean[key + "_mean"] = val / len(val_loader)

    avg_psnr /= len(val_loader)
    avg_lpips_alex /= len(val_loader)
    avg_lpips_vgg /= len(val_loader)
    avg_ssim /= len(val_loader)

    wandb.log({**psnr_mean, **lpips_alex_mean, **lpips_vgg_mean,
               **ssim_mean, "val/PSNR_epoch_mean": avg_psnr,
               "epoch_step":epoch_idx,
               "val/LPIPS_alexnet_epoch_mean": avg_lpips_alex,
               "val/LPIPS_vgg_epoch_mean": avg_lpips_vgg,
               "val/SSIM_epoch_mean": avg_ssim })




def check_frozen(ckpt, ref_ckpt, layer_res_list=None):
    eps = 0.0000001
    i=0
    for key in ckpt.keys():
        if ".0." in key:
            continue
        w = ckpt[key]
        ref_w = ref_ckpt[key]
        diff = w - ref_w

        if "weight" in key:
            if layer_res_list is not None:
                max_diff = (diff-layer_res_list[i]).abs().max()
                i += 1
            else:
                max_diff = diff.abs().max()


        if "bias" in key:
            max_diff = diff.abs().max()

        if max_diff > eps:
            logging.error(key + " was not the same\nmax diff: " + str(max_diff))
            return False
        if layer_res_list is not None and i == len(layer_res_list):
            break
    return True

@hydra.main(config_path="conf", config_name="config")
def main(args: DictConfig) -> None:
    # parser = argparse.ArgumentParser(description='shapenet few-shot view synthesis')
    # parser.add_argument('--config', type=str, required=True,
    #                     help='config file for the shape class (cars, chairs or lamps)')
    # parser.add_argument('--weight_path', type=str,default=None,
    #                     help='config file for the shape class (cars, chairs '
    #                          'or lamps)')
    # parser.add_argument('--debug_overfit_single_scene',  default=False,
    #                     action="store_true")
    # parser.add_argument('--use_reptile_loss', default=False,
    #                     action="store_true")
    # parser.add_argument("--note", type=str, default=None)
    # parser.add_argument("--std_scale", type=float, default=0.2)
    # args = parser.parse_args()

    # with open(args.config) as config:
    #     info = json.load(config)
    #     for key, value in info.items():
    #         args.__dict__[key] = value
    # args.savedir = Path(args.savedir)

    global cwd
    cwd = os.getcwd()

    wandb.init(name="train_"+args.exp_name, dir=cwd, project="meta_NeRF", entity="stereo",
               save_code=True, job_type="train")

    wandb.config.update(args)

    # device_idx = (HydraConfig.get().job.num) % torch.cuda.device_count()
    # device = torch.device("cuda:"+str(device_idx))
    device = torch.device("cuda")
    train_set = build_shapenetV2(args, image_set="train", dataset_root=args.data.dataset_root,
                                splits_path=args.data.splits_path, num_views=args.nerf.train_views)
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True)

    val_set = build_shapenetV2(args, image_set="val", dataset_root=args.data.dataset_root,
                            splits_path=args.data.splits_path,
                            num_views=args.nerf.tto_views+args.nerf.test_views)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

    nerf_model = build_nerf(args.nerf)
    nerf_model.to(device)

    out_channel = 0
    for l in nerf_model.net:
        if hasattr(l, "weight"):
            c = 1
            for x in l.weight.shape:
                c *= x
            out_channel += c

    gen_model = WeightGenerator(args, out_channel=out_channel)
    gen_model.to(device)
    gen_model.feature_extractor.to(device)

    gen_optim = torch.optim.Adam(gen_model.gen.parameters(), lr=args.nerf.meta_lr)

    global lpips_alex
    global lpips_vgg
    lpips_alex = lpips.LPIPS(net='alex').to(device)  # best forward scores
    lpips_vgg = lpips.LPIPS(
        net='vgg').to(device)  # closer to "traditional" perceptual loss, when used for optimization

    global val_views

    val_views = make_img_idx(args.nerf.test_views, args.feat.weight_gen_views, args.val_per_scene)

    logging.info("Training set: " + str(len(train_loader)) + " images" )
    logging.info("Val set: " + str(len(val_loader)) + " images")
    logging.info("validation views:\n" + str(val_views))

    if getattr(args.data, "weight_path", None) is not None:
        checkpoint = torch.load(args.data.weight_path, map_location=device)
        gen_model.load_state_dict(checkpoint['gen_model_state_dict'])
        gen_optim.load_state_dict(checkpoint['gen_optim_state_dict'])

    if args.feat.feature_extractor_type == "mvsnet":
        checkpoint = torch.load(args.feat.mvsnet_weight_path, map_location=device)
        gen_model.feature_extractor.load_state_dict(checkpoint["network_mvs_state_dict"])

    if getattr(args.data, "nerf_weight_path", None) is not None:
        nerf_checkpoint = torch.load(args.data.nerf_weight_path, map_location=device)

        if "nerf_model_state_dict" in nerf_checkpoint.keys():
            nerf_checkpoint = nerf_checkpoint["nerf_model_state_dict"]

        elif "meta_model_state_dict" in nerf_checkpoint.keys():
            nerf_checkpoint = nerf_checkpoint["meta_model_state_dict"]
        else:
            logging.error("checkpoint doesn't contain meta-nerf initialized weights")
            raise ValueError()
        nerf_model.load_state_dict(nerf_checkpoint)
    else:
        logging.error("must provide path to metaNeRF initial weights")
        raise ValueError()

    wandb.watch(gen_model.gen, log="all", log_freq=100)
    # val_meta(args, 0, nerf_model, gen_model, val_loader, device)
    for epoch in range(1, args.nerf.meta_epochs+1):
        logging.info("Epoch " + str(epoch) + " training...")
        train_meta(args, epoch, nerf_model, gen_model, gen_optim, train_loader, device, ref_state_dict=nerf_checkpoint)
        val_meta(args, epoch, nerf_model, gen_model, val_loader, device)

        ckpt_name = cwd+"/" + args.nerf.exp_name + "_epoch" + str(epoch) + ".pth"
        torch.save({
            'epoch': epoch,
            'gen_model_state_dict': gen_model.state_dict(),
            'gen_optim_state_dict': gen_optim.state_dict(),
            'nerf_model_state_dict': nerf_model.state_dict()
        }, ckpt_name)
        wandb.save(ckpt_name)
    logging.info("Testing...")
    test(args, nerf_model=nerf_model, gen_model=gen_model)
    logging.info("Complete!")


if __name__ == '__main__':
    main()