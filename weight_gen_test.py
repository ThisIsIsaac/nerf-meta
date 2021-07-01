from pathlib import Path
import argparse
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets.shapenetV2 import build_shapenetV2
from models.nerf import build_nerf, set_grad
from models.weight_generator import WeightGenerator, add_weight_res
from utils.shape_video import create_360_video
from models.rendering import get_rays_shapenet, sample_points, volume_render
import wandb
import copy

def test_time_optimize(args, model, optim, imgs, poses, hwf, bound):
    """
    test-time-optimize the meta trained model on available views
    """
    pixels = imgs.reshape(-1, 3)

    rays_o, rays_d = get_rays_shapenet(hwf, poses)
    rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)

    num_rays = rays_d.shape[0]
    for step in range(args.tto_steps):
        indices = torch.randint(num_rays, size=[args.tto_batchsize])
        raybatch_o, raybatch_d = rays_o[indices], rays_d[indices]
        pixelbatch = pixels[indices] 
        t_vals, xyz = sample_points(raybatch_o, raybatch_d, bound[0], bound[1],
                                    args.num_samples, perturb=True)
        
        optim.zero_grad()
        rgbs, sigmas = model(xyz)
        colors = volume_render(rgbs, sigmas, t_vals, white_bkgd=True)
        loss = F.mse_loss(colors, pixelbatch)
        loss.backward()
        optim.step()


def report_result(args, model, imgs, poses, hwf, bound):
    """
    report view-synthesis result on heldout views
    """
    ray_origins, ray_directions = get_rays_shapenet(hwf, poses)

    view_psnrs = []
    for img, rays_o, rays_d in zip(imgs, ray_origins, ray_directions):
        rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)
        t_vals, xyz = sample_points(rays_o, rays_d, bound[0], bound[1],
                                    args.num_samples, perturb=False)
        
        synth = []
        num_rays = rays_d.shape[0]
        with torch.no_grad():
            for i in range(0, num_rays, args.test_batchsize):
                rgbs_batch, sigmas_batch = model(xyz[i:i+args.test_batchsize])
                color_batch = volume_render(rgbs_batch, sigmas_batch,
                                            t_vals[i:i+args.test_batchsize],
                                            white_bkgd=True)
                synth.append(color_batch)
            synth = torch.cat(synth, dim=0).reshape_as(img)
            error = F.mse_loss(img, synth)
            psnr = -10*torch.log10(error)
            view_psnrs.append(psnr)
    
    scene_psnr = torch.stack(view_psnrs).mean()
    return scene_psnr


def test(args, nerf_model=None, gen_model=None, epoch_idx=1):
    print("testing...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_set = build_shapenetV2(args, image_set="test", dataset_root=args.dataset_root,
                            splits_path=args.splits_path,
                            num_views=args.tto_views+args.test_views)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
    test_nerf_model = copy.deepcopy(nerf_model)

    # model = build_nerf(args)
    # model.to(device)
    # wandb.save(args.weight_path)
    nerf_state = copy.deepcopy(test_nerf_model.state_dict())
    savedir = args.savedir
    # savedir = Path(args.savedir)
    savedir.mkdir(exist_ok=True)
    
    test_psnrs = []
    wandb_step = (epoch_idx-1)*len(test_loader)
    for idx, batch in enumerate(test_loader):
        imgs = batch["imgs"]
        poses = batch["poses"]
        hwf = batch["hwf"]
        bound = batch["bound"]
        imgs, poses, hwf, bound = imgs.to(device), poses.to(device), hwf.to(device), bound.to(device)
        imgs, poses, hwf, bound = imgs.squeeze(), poses.squeeze(), hwf.squeeze(), bound.squeeze()

        tto_imgs, test_imgs = torch.split(imgs, [args.tto_views, args.test_views], dim=0)
        tto_poses, test_poses = torch.split(poses, [args.tto_views, args.test_views], dim=0)
        tto_pixels = tto_imgs.reshape(-1, 3)
        rays_o, rays_d = get_rays_shapenet(hwf, tto_poses)
        rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)
        num_rays = rays_d.shape[0]


        wandb.log({'test/test_time_opt_imgs': wandb.Image(torch.squeeze(torch.transpose(tto_imgs, 0, 3)))},step=wandb_step)
        wandb.log({'test/test_time_input_imgs': wandb.Image(test_imgs.permute(0, 3, 1, 2))},step=wandb_step)

    
        test_nerf_model.load_state_dict(nerf_state)
        test_nerf_model = set_grad(test_nerf_model, False)

        with torch.no_grad():
            weight_res = gen_model(imgs)
            test_nerf_model = add_weight_res(test_nerf_model, weight_res,
                                       hidden_features=args.hidden_features,
                                       out_features=args.out_features)
            indices = torch.randint(num_rays, size=[args.train_batchsize])
            raybatch_o, raybatch_d = rays_o[indices], rays_d[indices]
            pixelbatch = tto_pixels[indices]
            t_vals, xyz = sample_points(raybatch_o, raybatch_d, bound[0],
                                        bound[1],
                                        args.num_samples, perturb=True)
            rgbs, sigmas = test_nerf_model(xyz)
            colors = volume_render(rgbs, sigmas, t_vals, white_bkgd=True)
            test_loss = F.mse_loss(colors, pixelbatch)

        inner_val_model = copy.deepcopy(test_nerf_model)
        inner_val_model = set_grad(inner_val_model, True)
        nerf_optim = torch.optim.SGD(inner_val_model.parameters(), args.tto_lr)

        #! function body of "test_time_optimize"

        has_recorded_without_tto = False

        for step in range(args.tto_steps):
            #* log output on every iteration
            if step % args.tto_log_steps == 0:
                if step == 0:
                    if has_recorded_without_tto == False:
                        with torch.no_grad():
                            scene_psnr = report_result(args, inner_val_model, test_imgs, test_poses, hwf,
                                                       bound)
                            wandb.log({"test/scene_psnr_step_" + str(step): scene_psnr}, step=wandb_step)

                            vid_frames = create_360_video(args, inner_val_model, hwf, bound, device,
                                                          idx + 1, savedir, step=step)
                            wandb.log({"test/vid_step_" + str(step): wandb.Video(
                                vid_frames.transpose(0, 3, 1, 2), fps=30, format="mp4")}, step=wandb_step)
                            has_recorded_without_tto = True

                else:
                    with torch.no_grad():
                        scene_psnr = report_result(args, inner_val_model, test_imgs,
                                                   test_poses, hwf,
                                                   bound)
                        wandb.log({"test/scene_psnr_step_" + str(step): scene_psnr}, step=wandb_step)

                        vid_frames = create_360_video(args, inner_val_model, hwf, bound,
                                                      device,
                                                      idx + 1, savedir, step=step)
                        wandb.log({"test/vid_step_" + str(step): wandb.Video(
                            vid_frames.transpose(0, 3, 1, 2), fps=30,
                            format="mp4")}, step=wandb_step)

            indices = torch.randint(num_rays, size=[args.tto_batchsize])
            raybatch_o, raybatch_d = rays_o[indices], rays_d[indices]
            pixelbatch = tto_pixels[indices]
            t_vals, xyz = sample_points(raybatch_o, raybatch_d, bound[0],
                                        bound[1],
                                        args.num_samples, perturb=True)

            nerf_optim.zero_grad()
            rgbs, sigmas = inner_val_model(xyz)
            colors = volume_render(rgbs, sigmas, t_vals, white_bkgd=True)
            loss = F.mse_loss(colors, pixelbatch)
            loss.backward()
            nerf_optim.step()

        #* log output of the very last iteration
        with torch.no_grad():
            scene_psnr = report_result(args, inner_val_model, test_imgs, test_poses, hwf,
                                       bound)
            vid_frames = create_360_video(args, inner_val_model, hwf, bound, device, idx + 1,
                                          savedir, step=args.tto_steps)
            wandb.log({"test/vid_step_" + str(args.tto_steps): wandb.Video(
                vid_frames.transpose(0, 3, 1, 2), fps=30, format="mp4")}, step=wandb_step)

        print(f"scene {idx+1}, psnr:{scene_psnr:.3f}, video created")
        wandb.log({"test/scene_psnr_" + str(args.tto_steps): scene_psnr}, step=wandb_step)
        test_psnrs.append(scene_psnr)
        wandb_step+=1
    
    test_psnrs = torch.stack(test_psnrs)
    psnr_mean = test_psnrs.mean()
    wandb.save(str(savedir))
    print("----------------------------------")
    print(f"test dataset mean psnr: " + str(psnr_mean))




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='shapenet few-shot view synthesis')
    parser.add_argument('--config', type=str, required=True,
                    help='config file for the shape class (cars, chairs or lamps)')
    parser.add_argument('--weight-path', type=str, required=True,
                        help='path to the meta-trained weight file')
    args = parser.parse_args()

    with open(args.config) as config:
        info = json.load(config)
        for key, value in info.items():
            args.__dict__[key] = value

    wandb.init(name=args.exp_name + "_test", dir="/root/nerf-meta-main/", project="meta_NeRF", entity="stereo",
               save_code=True, job_type="train")

    wandb.config.update(args)
    print("running test with weights at path: " + args.weight_path)
    test(args)