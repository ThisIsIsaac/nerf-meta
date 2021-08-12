import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from models.nerf import build_nerf
from torchvision import transforms
from models.mvsnerf import MVSNet
import wandb

class WeightGenerator(nn.Module):
    def __init__(self, args, out_channel):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        # source: https://github.com/yunjey/pytorch-tutorial/blob/0500d3df5a2a8080ccfccbc00aca0eacc21818db/tutorials/03-advanced/image_captioning/model.py#L9
        super(WeightGenerator, self).__init__()
        self.feature_extractor_type = args.feat.feature_extractor_type
        if args.feat.feature_extractor_type == " vgg":
            feature_extractor = models.vgg16_bn(pretrained=True)
            feature_extractor.requires_grad_(True)
            feature_extractor.train()
            self.feature_extractor = list(feature_extractor.children())[:-2][0]
            self.extraction_layers = [5, 12, 22, 32, 42]

            # Image preprocessing, normalization for the pretrained resnet
            # source: https://github.com/yunjey/pytorch-tutorial/blob/0500d3df5a2a8080ccfccbc00aca0eacc21818db/tutorials/03-advanced/image_captioning/train.py#L22
            # self.transform = transforms.Compose([
            #     transforms.Normalize((0.485, 0.456, 0.406),
            #                          (0.229, 0.224, 0.225))
            # ])
            self.compressor = nn.Sequential(
                nn.Conv3d(25, 8, kernel_size=7, stride=2, padding=3), nn.ReLU())
            self.in_channel = 753664

        elif args.feat.feature_extractor_type == "mvsnet":
            self.feature_extractor = MVSNet().requires_grad_(True)
            self.feature_extractor.train()
            # self.transform = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                                         std=[0.229, 0.224, 0.225]),
            #                             ])

            comp_in_channel = 8
            #* number of channel per voxel is doubled everytime the spatial dimensions (D x H x W) are halved.
            #* spatial dimensions are halved until 4x4x4
            self.compressor = nn.Sequential(
                nn.Conv3d(comp_in_channel, comp_in_channel*2, kernel_size=5, stride=2, padding=2), nn.ReLU(),
                nn.Conv3d(comp_in_channel*2, comp_in_channel*4, kernel_size=5,stride=2, padding=2), nn.ReLU(),
                nn.Conv3d(comp_in_channel*4, comp_in_channel * 8, kernel_size=5, stride=2,padding=2), nn.ReLU(),
                nn.Conv3d(comp_in_channel * 8, comp_in_channel * 16, kernel_size=5, stride=2,padding=2), nn.ReLU(),
                nn.Conv3d(comp_in_channel * 16, comp_in_channel * 16, kernel_size=3, stride=2,padding=1), nn.ReLU())
            self.in_channel = (comp_in_channel * 16) * 4 * 4 * 4

        self.num_layers = args.nerf.hidden_layers+1
        self.out_channel = out_channel
        self.hidden_channel=256

        self.gen = \
            nn.Sequential(
                nn.Linear(self.in_channel, self.hidden_channel), nn.ReLU(),
                nn.Linear(self.hidden_channel, self.hidden_channel), nn.ReLU(),
                nn.Linear(self.hidden_channel, self.hidden_channel), nn.ReLU(),
                nn.Linear(self.hidden_channel, self.hidden_channel), nn.ReLU(),
                nn.Linear(self.hidden_channel, self.hidden_channel), nn.ReLU(),
                nn.Linear(self.hidden_channel, self.hidden_channel), nn.ReLU(),
                nn.Linear(self.hidden_channel,self.out_channel))


    # pad=24 is used as the default value for dtu mvsnet
    def forward(self, imgs, proj_mats=None, near_fars=None,pad=24):
        """Extract feature vectors from input images."""
        # source: https://github.com/yunjey/pytorch-tutorial/blob/0500d3df5a2a8080ccfccbc00aca0eacc21818db/tutorials/03-advanced/image_captioning/model.py#L18
        imgs = imgs.permute(0, 3, 1, 2)
        # imgs = self.transform(imgs)

        if self.feature_extractor_type == "vgg":
            features=[]
            with torch.no_grad():
                feat=imgs
                for i in range(len(self.feature_extractor)):
                    feat = self.feature_extractor[i](feat)
                    if i in self.extraction_layers:
                        feat = F.interpolate(feat, size=[64, 64], mode="bilinear", align_corners=True)
                        features.append(feat)
                features = torch.cat(features, dim=1)
                features = torch.unsqueeze(features, 0)

        if self.feature_extractor_type == "mvsnet":
            features = self.feature_extractor(imgs, proj_mats, near_fars, pad=pad)[0]

        features = torch.squeeze(self.compressor(features))
        features = features.view(-1)
        weight_res = self.gen(features)
        return weight_res

def add_weight_res(nerf, res, hidden_layers=6, log_round=False, setup="train/",
                   std_scale=0.2):
    res *= 0.2
    logs = dict()
    idx=0
    for i in range(hidden_layers):
        l = i*2+1
        weight_shape = nerf.net[l].weight.shape
        c = 1
        for x in weight_shape:
            c *= x
        layer_res = torch.unsqueeze(res[idx:idx+c], 1)
        layer_res = layer_res.view(weight_shape)*std_scale
        idx+=c

        if log_round:
            logs[setup+"layer" + str(l) + "_weight_mean"] = torch.mean(nerf.net[l].weight.data)
            logs[setup+"layer" + str(l) + "_weight_std"] = torch.std(nerf.net[l].weight.data)
            logs[setup+"layer" + str(l) + "_res_mean"] = torch.mean(layer_res)
            logs[setup+"layer" + str(l) + "_res_std"] = torch.std(layer_res)


        #! https://discuss.pytorch.org/t/assign-parameters-to-nn-module-and-have-grad-fn-track-it/62677/2
        w = nerf.net[l].weight + (layer_res)
        del nerf.net[l].weight
        nerf.net[l].weight = torch.squeeze(w)


    return nerf, logs
