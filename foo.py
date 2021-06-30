import argparse
import json
import copy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets.shapenetV2 import build_shapenetV2
from models.weight_generator import WeightGenNerf
from models.rendering import get_rays_shapenet, sample_points, volume_render
from rich import print
from rich import pretty
pretty.install()
from rich import traceback
traceback.install()

def train_step(model, optim, imgs, poses, hwf, bound, rays_o, rays_d, num_samples, raybatch_size):
    pixels = imgs.reshape(-1, 3)
    num_rays = rays_d.shape[0]
    indices = torch.randint(num_rays, size=[raybatch_size])
    raybatch_o, raybatch_d = rays_o[indices], rays_d[indices]
    pixelbatch = pixels[indices]
    t_vals, xyz = sample_points(raybatch_o, raybatch_d, bound[0], bound[1],
                                num_samples, perturb=True)

    optim.zero_grad()
    rgbs, sigmas = model(imgs, xyz)
    colors = volume_render(rgbs, sigmas, t_vals, white_bkgd=True)
    loss = F.mse_loss(colors, pixelbatch)
    loss.backward()
    optim.step()

def report_result(model, input_imgs, input_poses, imgs, poses, hwf, bound, num_samples, raybatch_size):
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
                rgbs_batch, sigmas_batch = model(input_imgs, xyz[i:i + raybatch_size])
                color_batch = volume_render(rgbs_batch, sigmas_batch,
                                            t_vals[i:i + raybatch_size],
                                            white_bkgd=True)
                synth.append(color_batch)
            synth = torch.cat(synth, dim=0).reshape_as(img)
            error = F.mse_loss(img, synth)
            psnr = -10 * torch.log10(error)
            view_psnrs.append(psnr)

    scene_psnr = torch.stack(view_psnrs).mean()
    return scene_psnr

def val(args, model, val_loader, device):
    """
    validate the meta trained model for few-shot view synthesis
    """
    meta_trained_state = model.state_dict()
    val_model = copy.deepcopy(model)
    model.eval()

    val_psnrs = []
    val_losses=[]
    for imgs, poses, hwf, bound in val_loader:
        imgs, poses, hwf, bound = imgs.to(device), poses.to(device), hwf.to(
            device), bound.to(device)
        imgs, poses, hwf, bound = imgs.squeeze(), poses.squeeze(), \
                                  hwf.squeeze(), bound.squeeze()

        tto_imgs, test_imgs = torch.split(imgs,
                                          [args.tto_views, args.test_views],
                                          dim=0)
        tto_poses, test_poses = torch.split(poses,
                                            [args.tto_views, args.test_views],
                                            dim=0)

        val_model.load_state_dict(meta_trained_state)

        pixels = imgs.reshape(-1, 3)

        rays_o, rays_d = get_rays_shapenet(hwf, poses)
        rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)

        num_rays = rays_d.shape[0]

        indices = torch.randint(num_rays, size=[raybatch_size])
        raybatch_o, raybatch_d = rays_o[indices], rays_d[indices]
        pixelbatch = pixels[indices]
        t_vals, xyz = sample_points(raybatch_o, raybatch_d, bound[0],
                                    bound[1],
                                    args.num_samples, perturb=True)

        with torch.no_grad():
            rgbs, sigmas = model(imgs, xyz)

        colors = volume_render(rgbs, sigmas, t_vals, white_bkgd=True)
        val_loss = F.mse_loss(colors, pixelbatch)

        scene_psnr = report_result(val_model, tto_imgs, tto_poses, test_imgs, test_poses, hwf, bound,
                                   args.num_samples, args.test_batchsize)
        val_psnrs.append(scene_psnr)
        val_losses.append(val_loss)

    val_psnr = torch.stack(val_psnrs).mean()
    val_loss = torch.stack(val_losses).mean()
    return val_psnr, val_loss


def main():
    parser = argparse.ArgumentParser(
        description='shapenet few-shot view synthesis')
    parser.add_argument('--config', type=str, required=True,
                        help='config file for the shape class (cars, chairs '
                             'or lamps)')
    parser.add_argument('--weight_path', type=str, default=None)
    args = parser.parse_args()

    with open(args.config) as config:
        info = json.load(config)
        for key, value in info.items():
            args.__dict__[key] = value

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_set = build_shapenetV2(args=args, image_set="train",
                               dataset_root=args.dataset_root,
                               splits_path=args.splits_path,
                               num_views=args.train_views)
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True)

    val_set = build_shapenetV2(args=args, image_set="val", dataset_root=args.dataset_root,
                             splits_path=args.splits_path,
                             num_views=args.tto_views + args.test_views)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

    model = WeightGenNerf(args)
    model.to(device)

    if args.weight_path is not None:
        checkpoint = torch.load(args.weight_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

    elif args.nerf_weight_path is not None:
        checkpoint = torch.load(args.nerf_weight_path, map_location=device)
        meta_state = checkpoint['meta_model_state_dict']
        model.nerf.load_state_dict(meta_state)

    else:
        print("must provide path to metaNeRF initial weights")

    #! is this the right way to stop gradient? it seems to be making extra copy unnecessarily
    # model.nerf = model.nerf.detach()
    model.nerf.requires_grad = False

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        for batch in train_loader:
            imgs = batch["imgs"]
            poses = batch["poses"]
            hwf = batch["hwf"]
            bound = batch["bound"]
            rays_o = batch["rays_o"]
            rays_d = batch["rays_d"]

            imgs, poses, hwf, bound = imgs.squeeze(), poses.squeeze(), \
                                      hwf.squeeze(), bound.squeeze()

            train_step(model, optim, imgs, poses, hwf, bound, rays_o, rays_d,
                       args.num_samples, args.batchsize)


        val_psnr = val(args, model, val_loader, device)
        print(f"Epoch: {epoch}, val psnr: {val_psnr:0.3f}")

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optim_state_dict': optim.state_dict(),
        }, f'meta_epoch{epoch}.pth')


if __name__ == '__main__':
    main()