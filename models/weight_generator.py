import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from models.nerf import build_nerf
from torchvision import transforms

class WeightGenerator(nn.Module):
    def __init__(self, args):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        # source: https://github.com/yunjey/pytorch-tutorial/blob/0500d3df5a2a8080ccfccbc00aca0eacc21818db/tutorials/03-advanced/image_captioning/model.py#L9
        super(WeightGenerator, self).__init__()
        feature_extractor = models.vgg16_bn(pretrained=True)
        feature_extractor.requires_grad_(False)
        feature_extractor.eval()
        self.feature_extractor = list(feature_extractor.children())[:-2][0]
        self.extraction_layers = [5, 12, 22, 32, 42]

        self.hidden_features = args.hidden_features
        self.hidden_layers = args.hidden_layers
        self.out_features = args.out_features
        self.gen = nn.Sequential(nn.Linear(1472,args.hidden_layers), nn.ReLU())

        # Image preprocessing, normalization for the pretrained resnet
        # source: https://github.com/yunjey/pytorch-tutorial/blob/0500d3df5a2a8080ccfccbc00aca0eacc21818db/tutorials/03-advanced/image_captioning/train.py#L22
        self.transform = transforms.Compose([
            # transforms.ToTensor(),
            # transforms.Resize((args.img_resize_h, args.img_resize_w)),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))
        ])

    def forward(self, imgs):
        """Extract feature vectors from input images."""
        # source: https://github.com/yunjey/pytorch-tutorial/blob/0500d3df5a2a8080ccfccbc00aca0eacc21818db/tutorials/03-advanced/image_captioning/model.py#L18
        imgs = imgs.permute(0, 3, 1, 2)
        imgs = self.transform(imgs)

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
        weight_res = self.gen(features)
        weight_res = weight_res.permute(0, 3, 1, 2)
        weight_res = F.interpolate(weight_res, size=[256, 256],mode="bilinear", align_corners=True)
        return weight_res

def add_weight_res(nerf, res, hidden_features=256, out_features=4, debug=False):
    layers_before_fc = 3
    _, _, res_H, res_W = res.shape



    layer_res_list = []
    num_layers = (len(nerf.net)-layers_before_fc)//2
    for i in range(num_layers):
        # weight_std = torch.std(nerf.net[layers_before_fc + 2 * i].weight, 0)
        # res_std = torch.std(res[:, i, :, :], 0)
        # zero_mask = res_std==0
        #
        # scale =
        l = i*2+1
        weight_shape = nerf.net[l].weight.shape
        layer_res = torch.unsqueeze(res[:, i, :, :], 1)
        layer_res = F.interpolate(layer_res, weight_shape, mode="bilinear", align_corners=True)
        layer_res = torch.squeeze(torch.mean(layer_res, dim=0))
        nerf.net[l].weight.data += layer_res
        layer_res_list.append(layer_res)
    if debug:
        return nerf, layer_res_list

    return nerf
