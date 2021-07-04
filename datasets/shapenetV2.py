from pathlib import Path
import json
import imageio
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from models.rendering import get_rays_shapenet, sample_points, volume_render

class ShapenetDatasetV2(Dataset):
    """
    returns the images, poses and instrinsics of a partiucular scene
    """

    def __init__(self, args, all_folders, num_views):
        """
        Args:
            all_folders (list): list of folder paths. each folder contains
            indiviual scene info
            num_views (int): number of views to return for each scene
        """
        super().__init__()
        self.all_folders = all_folders
        self.num_views = num_views


    def __getitem__(self, idx):
        folderpath = self.all_folders[idx]
        meta_path = folderpath.joinpath("transforms.json")
        with open(meta_path, "r") as meta_file:
            meta_data = json.load(meta_file)

        imgs = []
        poses = []
        for frame_idx in range(self.num_views):
            frame = meta_data["frames"][frame_idx]

            img_name = f"{Path(frame['file_path']).stem}.png"
            img_path = folderpath.joinpath(img_name)
            img = imageio.imread(img_path)
            imgs.append(torch.as_tensor(img, dtype=torch.float))

            pose = frame["transform_matrix"]
            poses.append(torch.as_tensor(pose, dtype=torch.float))

        imgs = torch.stack(imgs, dim=0) / 255.
        # composite the images to a white background
        imgs = imgs[..., :3] * imgs[..., -1:] + 1 - imgs[...,
                                                    -1:]

        poses = torch.stack(poses, dim=0)

        # all images of a scene has the same camera intrinsics
        H, W = imgs[0].shape[:2]
        camera_angle_x = meta_data["camera_angle_x"]
        camera_angle_x = torch.as_tensor(camera_angle_x, dtype=torch.float)

        # camera angle equation: tan(angle/2) = (W/2)/focal
        focal = 0.5 * W / torch.tan(0.5 * camera_angle_x)
        hwf = torch.as_tensor([H, W, focal], dtype=torch.float)

        # all shapenet scenes are bounded between 2. and 6.
        near = 2.
        far = 6.
        bound = torch.as_tensor([near, far], dtype=torch.float)

        return {
            "imgs": imgs,
            "poses": poses,
            "hwf":hwf,
            "bound":bound,
        }


    def __len__(self):
        return len(self.all_folders)


def build_shapenetV2(args, image_set, dataset_root, splits_path, num_views):
    """
    Args:
        image_set: specifies whether to return "train", "val" or "test" dataset
        dataset_root: root path of the dataset
        splits_path: file path that specifies train, val and test split
        num_views: num of views to return from a single scene
    """
    root_path = Path(dataset_root)
    splits_path = Path(splits_path)
    with open(splits_path, "r") as splits_file:
        splits = json.load(splits_file)

    all_folders = [root_path.joinpath(foldername) for foldername in
                   sorted(splits[image_set])]
    dataset = ShapenetDatasetV2(args, all_folders, num_views)

    return dataset