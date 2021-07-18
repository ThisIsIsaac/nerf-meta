import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from models.nerf import build_nerf
from torchvision import transforms
from models.mvsnerf import MVSNet
import matplotlib.pyplot as plt
import wandb

class WeightGenerator(nn.Module):
    def __init__(self, args):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        # source: https://github.com/yunjey/pytorch-tutorial/blob/0500d3df5a2a8080ccfccbc00aca0eacc21818db/tutorials/03-advanced/image_captioning/model.py#L9
        super(WeightGenerator, self).__init__()
        self.feature_extractor_type = args.feature_extractor_type
        if args.feature_extractor_type == "resnet":
            feature_extractor = models.vgg16_bn(pretrained=True)
            feature_extractor.requires_grad_(False)
            feature_extractor.eval()
            self.feature_extractor = list(feature_extractor.children())[:-2][0]
            self.extraction_layers = [5, 12, 22, 32, 42]
            # Image preprocessing, normalization for the pretrained resnet
            # source: https://github.com/yunjey/pytorch-tutorial/blob/0500d3df5a2a8080ccfccbc00aca0eacc21818db/tutorials/03-advanced/image_captioning/train.py#L22

            self.transform = transforms.Compose([
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))
            ])

        elif args.feature_extractor_type == "mvsnet":
            self.feature_extractor = MVSNet().requires_grad_(False)
            self.feature_extractor.eval()
            self.transform=lambda x: x

            self.compressor = nn.Sequential(nn.Linear(8, 1), nn.ReLU())


        # self.hidden_features = args.hidden_features
        self.num_layers = args.hidden_layers+1
        self.out_features = args.out_features
        self.in_channel = 1472 if self.feature_extractor_type == "resnet" else 128
        self.gen = \
            nn.Sequential(
                nn.Linear(self.in_channel, 128), nn.ReLU(),
                nn.Linear(128, 128), nn.ReLU(),
                nn.Linear(128, 128), nn.ReLU(),
                nn.Linear(128, 128), nn.ReLU(),
                nn.Linear(128, 128), nn.ReLU(),
                nn.Linear(128, 128), nn.ReLU(),
                nn.Linear(128,self.num_layers))


    # pad=24 is used as the default value for dtu mvsnet
    def forward(self, imgs, proj_mats=None, near_fars=None,pad=24):
        """Extract feature vectors from input images."""
        # source: https://github.com/yunjey/pytorch-tutorial/blob/0500d3df5a2a8080ccfccbc00aca0eacc21818db/tutorials/03-advanced/image_captioning/model.py#L18
        imgs = imgs.permute(0, 3, 1, 2)
        imgs = self.transform(imgs)

        if self.feature_extractor_type == "resnet":
            features=[]
            with torch.no_grad():
                feat=imgs
                for i in range(len(self.feature_extractor)):
                    feat = self.feature_extractor[i](feat)
                    if i in self.extraction_layers:
                        feat = F.interpolate(feat, size=[64, 64], mode="bilinear", align_corners=True)
                        features.append(feat)
                features = torch.cat(features, dim=1)
                features = features.permute(0, 2, 3, 1)
                features = torch.squeeze(features)

        if self.feature_extractor_type == "mvsnet":
            features = self.feature_extractor(imgs, proj_mats, near_fars, pad=pad)
            features = torch.squeeze(features[0])
            features = features.permute(1, 2, 3, 0)
            features = torch.squeeze(self.compressor(features))
            features = features.permute(1, 2, 0)

        weight_res = self.gen(features)
        weight_res = torch.unsqueeze(weight_res.permute(2, 0, 1), 0)
        weight_res = F.interpolate(weight_res, size=[256, 256],mode="bilinear", align_corners=True)
        return weight_res

def add_weight_res(nerf, res, hidden_layers=5, log_round=False):
    _, _, res_H, res_W = res.shape
    res *= 0.2
    logs = dict()
    for i in range(hidden_layers):
        l = i*2+1
        weight_shape = nerf.net[l].weight.shape
        layer_res = torch.unsqueeze(res[:, i, :, :], 1)
        layer_res = F.interpolate(layer_res, weight_shape, mode="bilinear", align_corners=True)

        if log_round:
            # logs["layer" + str(l)] = wandb.Histogram(torch.flatten(nerf.net[l].weight.data.clone().detach().cpu()).numpy())
            # logs["res" + str(l)] = wandb.Histogram(torch.flatten(layer_res.clone().detach().cpu()).numpy())
            logs["layer" + str(l) + "_weight_mean"] = torch.mean(nerf.net[l].weight.data)
            logs["layer" + str(l) + "_weight_std"] = torch.std(nerf.net[l].weight.data)
            logs["layer" + str(l) + "_res_mean"] = torch.mean(layer_res)
            logs["layer" + str(l) + "_res_std"] = torch.std(layer_res)


        #! https://discuss.pytorch.org/t/assign-parameters-to-nn-module-and-have-grad-fn-track-it/62677/2
        w = nerf.net[l].weight + (layer_res)
        del nerf.net[l].weight
        nerf.net[l].weight = torch.squeeze(w)

        #! create another NeRF datastructure, assign w to the new datastructure and return the new datastructure.

        # layer_res_list.append(layer_res)

    return nerf, logs
