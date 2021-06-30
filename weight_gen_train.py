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


def inner_loop(nerf_model, nerf_optim, pixels, rays_o, rays_d, bound, num_samples, raybatch_size, inner_steps):
    """
    train the inner model for a specified number of iterations
    """
    # nerf_model.train()
    # nerf_model = set_grad(nerf_model, True)
    # pixels = imgs.reshrequires_gradape(-1, 3)

    # rays_o, rays_d = get_rays_shapenet(hwf, poses)
    # rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)

    num_rays = rays_d.shape[0]
    for step in range(inner_steps):
        indices = torch.randint(num_rays, size=[raybatch_size])
        raybatch_o, raybatch_d = rays_o[indices], rays_d[indices]
        pixelbatch = pixels[indices] 
        t_vals, xyz = sample_points(raybatch_o, raybatch_d, bound[0], bound[1],
                                    num_samples, perturb=True)
        
        nerf_optim.zero_grad()
        rgbs, sigmas = nerf_model(xyz)
        colors = volume_render(rgbs, sigmas, t_vals, white_bkgd=True)
        loss = F.mse_loss(colors, pixelbatch)
        loss.backward()
        nerf_optim.step()


def train_meta(args, nerf_model, gen_model, gen_optim, data_loader, device):
    """
    train the meta_model for one epoch using reptile meta learning
    https://arxiv.org/abs/1803.02999
    """
    gen_model.train()
    gen_model.requires_grad=True
    for batch in data_loader:
        imgs = batch["imgs"]
        poses = batch["poses"]
        hwf = batch["hwf"]
        bound = batch["bound"]
        # rays_o = batch["rays_o"]
        # rays_d = batch["rays_d"]
        # num_rays = rays_d.shape[1]

        imgs, poses, hwf, bound = imgs.to(device), poses.to(device), hwf.to(device), bound.to(device)
        imgs, poses, hwf, bound = imgs.squeeze(), poses.squeeze(), hwf.squeeze(), bound.squeeze()
        rays_o, rays_d = get_rays_shapenet(hwf, poses)
        rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)
        num_rays = rays_d.shape[0]
        pixels = imgs.reshape(-1, 3)
        # Train weight generator
        gen_optim.zero_grad()
        nerf_model_copy = copy.deepcopy(nerf_model) #! copy meta model initialized weights
        nerf_model_copy = set_grad(nerf_model_copy, False)     #! then turn gradient off
        weight_res = gen_model(imgs)
        add_weight_res(nerf_model_copy, weight_res, hidden_features=args.hidden_features, out_features=args.out_features)
        indices = torch.randint(num_rays, size=[args.train_batchsize])
        raybatch_o, raybatch_d = rays_o[indices], rays_d[indices]
        pixelbatch = pixels[indices]
        t_vals, xyz = sample_points(raybatch_o, raybatch_d, bound[0], bound[1],
                                    args.num_samples, perturb=True)
        rgbs, sigmas = nerf_model_copy(xyz)
        colors = volume_render(rgbs, sigmas, t_vals, white_bkgd=True)
        loss = F.mse_loss(colors, pixelbatch)
        loss.backward()
        gen_optim.step()

        inner_nerf_model_copy = copy.deepcopy(nerf_model_copy)
        inner_nerf_model_copy = set_grad(inner_nerf_model_copy, True)
        inner_nerf_model_copy.train()
        nerf_optim = torch.optim.SGD(inner_nerf_model_copy.parameters(), args.inner_lr)


        inner_loop(inner_nerf_model_copy, nerf_optim, pixels,
                    rays_o, rays_d, bound, args.num_samples,
                    args.train_batchsize, args.inner_steps)


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


def val_meta(args, nerf_model, gen_model, val_loader, device):
    """
    validate the meta trained model for few-shot view synthesis
    """
    gen_model.eval()
    gen_model.requires_grad=False
    meta_trained_state = nerf_model.state_dict()
    val_model = copy.deepcopy(nerf_model)

    val_psnrs_fin = []
    val_psnrs_0 = []
    for batch in val_loader:
        imgs = batch["imgs"]
        poses = batch["poses"]
        hwf = batch["hwf"]
        bound = batch["bound"]

        imgs, poses, hwf, bound = imgs.to(device), poses.to(device), hwf.to(device), bound.to(device)
        imgs, poses, hwf, bound = imgs.squeeze(), poses.squeeze(), hwf.squeeze(), bound.squeeze()

        rays_o, rays_d = get_rays_shapenet(hwf, poses)
        rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)
        num_rays = rays_d.shape[0]
        tto_imgs, test_imgs = torch.split(imgs, [args.tto_views, args.test_views], dim=0)
        tto_poses, test_poses = torch.split(poses, [args.tto_views, args.test_views], dim=0)

        tto_pixels = tto_imgs.reshape(-1, 3)
        # Add weight residual
        val_model.load_state_dict(meta_trained_state)
        val_model = set_grad(val_model, False)

        # val_model_copy = copy.deepcopy(nerf_model.detach()) #! copy and detach the original meta model so gradient doesn't flow to the initialized weights
        with torch.no_grad:
            weight_res = gen_model(imgs)
            val_model = add_weight_res(val_model, weight_res, hidden_features=args.hidden_features, out_features=args.out_features)
            indices = torch.randint(num_rays, size=[args.train_batchsize])
            raybatch_o, raybatch_d = rays_o[indices], rays_d[indices]
            pixelbatch = tto_pixels[indices]
            t_vals, xyz = sample_points(raybatch_o, raybatch_d, bound[0], bound[1],
                                        args.num_samples, perturb=True)
            rgbs, sigmas = val_model(xyz)
            colors = volume_render(rgbs, sigmas, t_vals, white_bkgd=True)
            val_loss = F.mse_loss(colors, pixelbatch)

        inner_val_model = copy.deepcopy(val_model)
        inner_val_model = set_grad(inner_val_model, True)
        val_optim = torch.optim.SGD(inner_val_model.parameters(), args.tto_lr)
        scene_psnr_0 = report_result(inner_val_model, test_imgs, test_poses, hwf, bound,
                                    args.num_samples, args.test_batchsize)

        inner_loop(inner_val_model, val_optim, tto_pixels, rays_o,
                    rays_d, bound, args.num_samples, args.tto_batchsize, args.tto_steps)
        
        scene_psnr_fin = report_result(inner_val_model, test_imgs, test_poses, hwf, bound,
                                    args.num_samples, args.test_batchsize)
        val_psnrs_0.append(scene_psnr_0)
        val_psnrs_fin.append(scene_psnr_fin)

    val_psnr_fin = torch.stack(val_psnrs_fin).mean()
    val_psnr_0 = torch.stack(val_psnrs_0).mean()
    return [val_psnr_0,val_psnr_fin]


def main():
    parser = argparse.ArgumentParser(description='shapenet few-shot view synthesis')
    parser.add_argument('--config', type=str, required=True,
                        help='config file for the shape class (cars, chairs or lamps)')
    parser.add_argument('--weight_path', type=str,default=None,
                        help='config file for the shape class (cars, chairs '
                             'or lamps)')
    args = parser.parse_args()

    with open(args.config) as config:
        info = json.load(config)
        for key, value in info.items():
            args.__dict__[key] = value

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
    gen_optim = torch.optim.Adam(gen_model.parameters(), lr=args.meta_lr)

    if args.weight_path is not None:
        checkpoint = torch.load(args.weight_path, map_location=device)
        gen_model.load_state_dict(checkpoint['gen_model_state_dict'])
        gen_optim.load_state_dict(checkpoint['gen_optim_state_dict'])

    if args.nerf_weight_path is not None:
        checkpoint = torch.load(args.nerf_weight_path, map_location=device)

        if "nerf_model_state_dict" in checkpoint.keys():
            nerf_model.load_state_dict(checkpoint["nerf_model_state_dict"])
        elif "meta_model_state_dict" in checkpoint.keys():
            nerf_model.load_state_dict(checkpoint["meta_model_state_dict"])
        else:
            print("checkpoint doesn't contain meta-nerf initialized weights")
            raise ValueError()
    else:
        print("must provide path to metaNeRF initial weights")
        raise ValueError()

    #! Delete me
    test(args, nerf_model=nerf_model, gen_model=gen_model)

    print("starting to train...")
    for epoch in range(1, args.meta_epochs+1):
        if epoch > 1:
            ckpt_name = "./" + args.exp_name + "_epoch" + str(epoch) + ".pth"
            torch.save({
                'epoch': epoch,
                'gen_model_state_dict': gen_model.state_dict(),
                'gen_optim_state_dict': gen_optim.state_dict(),
                'nerf_model_state_dict': nerf_model.state_dict()
            }, ckpt_name)
            wandb.save(ckpt_name)
            # args.weight_path = ckpt_name

        train_meta(args, nerf_model, gen_model, gen_optim, train_loader, device)
        [val_psnr_0, val_psnr_fin] = val_meta(args, nerf_model, val_loader, device)

        print(f"Epoch: {epoch}, val_psnr_0: {val_psnr_0:0.3f}")
        wandb.log({"epoch":epoch, "val_psnr_0": val_psnr_0})
        print(f"Epoch: {epoch}, val psnr fin: {val_psnr_fin:0.3f}")
        wandb.log({"epoch":epoch, "val_psnr_fin": val_psnr_fin})
    test(args, nerf_model=nerf_model, gen_model=gen_model)

if __name__ == '__main__':
    main()