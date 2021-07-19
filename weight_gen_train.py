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
def inner_loop(args, nerf_model, nerf_optim, pixels, imgs, rays_o, rays_d,
               poses, bound, hwf, num_samples, raybatch_size, inner_steps,
               device, idx, log_round=False, setup="train/"):
    """
    train the inner model for a specified number of iterations
    """
    num_rays = rays_d.shape[0]
    logs = dict()
    for i in range(1, inner_steps+1):
        if log_round and ((i % args.tto_log_steps == 0) or (i == inner_steps) or (i==1)):
            with torch.no_grad():
                scene_psnr = report_result(nerf_model, imgs,
                                           poses, hwf,
                                           bound, num_samples, raybatch_size)

                vid_frames = create_360_video(args, nerf_model, hwf, bound,
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
        

        rgbs, sigmas = nerf_model(xyz)
        colors = volume_render(rgbs, sigmas, t_vals, white_bkgd=True)
        loss = F.mse_loss(colors, pixelbatch)
        loss.backward()
        nerf_optim.step()
        nerf_optim.zero_grad()

        # if check_frozen(nerf_model.state_dict(), nerf_weight_orig ):
        #     print("Gradients of nerf model not updated during inner_step")

    return logs



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

        nerf_model_copy, logs_weight_stat = add_weight_res(nerf_model_copy, weight_res, log_round=log_round, setup="train/")
        indices = torch.randint(num_rays, size=[args.train_batchsize])
        raybatch_o, raybatch_d = rays_o[indices], rays_d[indices]
        pixelbatch = pixels[indices]
        t_vals, xyz = sample_points(raybatch_o, raybatch_d, bound[0], bound[1],
                                    args.num_samples, perturb=True)

        rgbs, sigmas = nerf_model_copy(xyz)
        colors = volume_render(rgbs, sigmas, t_vals, white_bkgd=True)

        loss = F.mse_loss(colors, pixelbatch)#.requires_grad_()
        loss.backward()
        gen_optim.step()
        gen_optim.zero_grad()

        inner_nerf_model_copy = copy.deepcopy(nerf_model)
        for i in range(len(nerf_model_copy.net)):
            layer = nerf_model_copy.net[i]
            if hasattr(layer, "weight"):
                inner_nerf_model_copy.net[i].weight = nn.Parameter(layer.weight)

        inner_nerf_model_copy = set_grad(inner_nerf_model_copy, True)
        inner_nerf_model_copy.train()
        nerf_optim = torch.optim.SGD(inner_nerf_model_copy.parameters(), args.inner_lr)

        logs = inner_loop(args, inner_nerf_model_copy, nerf_optim, pixels, imgs,
                    rays_o, rays_d, poses, bound, hwf, args.num_samples,
                    args.train_batchsize, args.inner_steps,
                    device=device, idx=idx, log_round=log_round, setup="train/")
        if log_round:
            logs["train/gen_model_mse_loss"] = float(loss)
            wandb.log({**logs, **logs_weight_stat, "train_step": train_step,
                       "train/imgs":wandb.Image(make_grid(imgs.permute(0, 3, 1, 2)))})
        step+=1

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


def val_meta(args, epoch_idx, nerf_model, gen_model, val_loader, device):
    """
    validate the meta trained model for few-shot view synthesis
    """
    gen_model.eval()
    gen_model.requires_grad_(False)
    meta_trained_state = nerf_model.state_dict()
    avg_psnr = 0
    val_step = (epoch_idx-1)*len(val_loader) +1
    for idx, batch in enumerate(val_loader):
        imgs = batch["imgs"]
        poses = batch["poses"]
        hwf = batch["hwf"]
        bound = batch["bound"]
        relative_poses = batch["relative_poses"]

        imgs, poses, hwf, bound, relative_poses = imgs.to(device), poses.to(
            device), hwf.to(device), bound.to(device), relative_poses.to(device)
        imgs, poses, hwf, bound, relative_poses = imgs.squeeze(), \
                                                  poses.squeeze(), \
                                                  hwf.squeeze(), \
                                                  bound.squeeze(), relative_poses.squeeze()

        tto_imgs, test_imgs = torch.split(imgs, [args.tto_views, args.test_views], dim=0)
        tto_poses, test_poses = torch.split(poses, [args.tto_views, args.test_views], dim=0)

        rays_o, rays_d = get_rays_shapenet(hwf, tto_poses)
        rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)
        num_rays = rays_d.shape[0]

        tto_pixels = tto_imgs.reshape(-1, 3)
        # Add weight residual
        val_model = copy.deepcopy(nerf_model)
        val_model.load_state_dict(meta_trained_state)
        val_model = set_grad(val_model, False)

        with torch.no_grad():
            weight_res = gen_model(imgs, relative_poses, bound)
            val_model, logs_weight_stat = add_weight_res(val_model, weight_res,
                                                         log_round=True, setup="val/")
            indices = torch.randint(num_rays, size=[args.train_batchsize])
            raybatch_o, raybatch_d = rays_o[indices], rays_d[indices]
            pixelbatch = tto_pixels[indices]
            t_vals, xyz = sample_points(raybatch_o, raybatch_d, bound[0], bound[1],
                                        args.num_samples, perturb=True)
            rgbs, sigmas = val_model(xyz)
            colors = volume_render(rgbs, sigmas, t_vals, white_bkgd=True)
            val_loss = F.mse_loss(colors, pixelbatch)


        inner_val_model = copy.deepcopy(val_model)
        for i in range(len(val_model.net)):
            layer = val_model.net[i]
            if hasattr(layer, "weight"):
                inner_val_model.net[i].weight = nn.Parameter(layer.weight)

        inner_val_model = set_grad(inner_val_model, True)
        inner_val_model.train()
        val_optim = torch.optim.SGD(inner_val_model.parameters(), args.tto_lr)


        logs = inner_loop(args, inner_val_model, val_optim, tto_pixels, tto_imgs, rays_o,
                    rays_d, tto_poses, bound, hwf, args.num_samples,
                   args.tto_batchsize, args.tto_steps,
                   device=device, idx=idx, log_round=True, setup="val/")

        avg_psnr += logs["val/scene_psnr tto_step=" + str(args.tto_steps)]
        logs["val/tto_views"] = wandb.Image(make_grid(tto_imgs.permute(0, 3, 1, 2)))
        logs["val/test_views"] = wandb.Image(make_grid(test_imgs.permute(0, 3, 1, 2)))
        logs["val/mse_loss"] = val_loss
        wandb.log({**logs, **logs_weight_stat, "val_step":val_step})
        val_step+=1
    avg_psnr /= len(val_loader)
    wandb.log({"val/avg_psnr":avg_psnr, "epoch_step":epoch_idx})




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
            print(key + " was not the same")
            print("max diff: " + str(max_diff))
            return False
        if layer_res_list is not None and i == len(layer_res_list):
            break
    return True


def main():
    parser = argparse.ArgumentParser(description='shapenet few-shot view synthesis')
    parser.add_argument('--config', type=str, required=True,
                        help='config file for the shape class (cars, chairs or lamps)')
    parser.add_argument('--weight_path', type=str,default=None,
                        help='config file for the shape class (cars, chairs '
                             'or lamps)')
    parser.add_argument('--debug_overfit_single_scene',  default=False,
                        action="store_true")
    args = parser.parse_args()

    with open(args.config) as config:
        info = json.load(config)
        for key, value in info.items():
            args.__dict__[key] = value
    args.savedir = Path(args.savedir)
    wandb.init(name="train_"+args.exp_name, dir="/root/nerf-meta-main/", project="meta_NeRF", entity="stereo",
               save_code=True, job_type="train")

    wandb.config.update(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_set = build_shapenetV2(args, image_set="train", dataset_root=args.dataset_root,
                                splits_path=args.splits_path, num_views=args.train_views)
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True)

    val_set = build_shapenetV2(args, image_set="val", dataset_root=args.dataset_root,
                            splits_path=args.splits_path,
                            num_views=args.tto_views+args.test_views)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

    gen_model = WeightGenerator(args)
    gen_model.to(device)
    gen_model.feature_extractor.to(device)
    nerf_model = build_nerf(args)
    nerf_model.to(device)
    gen_optim = torch.optim.Adam(gen_model.gen.parameters(), lr=args.meta_lr)

    print("Training set: " + str(len(train_loader)) + " images" )
    print("Val set: " + str(len(val_loader)) + " images")


    if args.weight_path is not None:
        checkpoint = torch.load(args.weight_path, map_location=device)
        gen_model.load_state_dict(checkpoint['gen_model_state_dict'])
        gen_optim.load_state_dict(checkpoint['gen_optim_state_dict'])

    if args.feature_extractor_type == "mvsnet":
        checkpoint = torch.load(args.mvsnet_weight_path, map_location=device)
        gen_model.feature_extractor.load_state_dict(checkpoint["network_mvs_state_dict"])

    if args.nerf_weight_path is not None:
        nerf_checkpoint = torch.load(args.nerf_weight_path, map_location=device)

        if "nerf_model_state_dict" in nerf_checkpoint.keys():
            nerf_checkpoint = nerf_checkpoint["nerf_model_state_dict"]

        elif "meta_model_state_dict" in nerf_checkpoint.keys():
            nerf_checkpoint = nerf_checkpoint["meta_model_state_dict"]
        else:
            print("checkpoint doesn't contain meta-nerf initialized weights")
            raise ValueError()
        nerf_model.load_state_dict(nerf_checkpoint)
    else:
        print("must provide path to metaNeRF initial weights")
        raise ValueError()

    wandb.watch(gen_model.gen, log="all", log_freq=100)
    for epoch in range(1, args.meta_epochs+1):
        print("Epoch " + str(epoch) + " training...")
        train_meta(args, epoch, nerf_model, gen_model, gen_optim, train_loader, device, ref_state_dict=nerf_checkpoint)
        val_meta(args, epoch, nerf_model, gen_model, val_loader, device)

        ckpt_name = "./" + args.exp_name + "_epoch" + str(epoch) + ".pth"
        torch.save({
            'epoch': epoch,
            'gen_model_state_dict': gen_model.state_dict(),
            'gen_optim_state_dict': gen_optim.state_dict(),
            'nerf_model_state_dict': nerf_model.state_dict()
        }, ckpt_name)
        wandb.save(ckpt_name)
    print("Testing...")
    test(args, nerf_model=nerf_model, gen_model=gen_model)
    print("Complete!")


if __name__ == '__main__':
    main()