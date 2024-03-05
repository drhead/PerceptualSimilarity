from collections import namedtuple
import torch
from torchvision import models as tv
import torch.nn.functional as F
from functools import reduce
import math

class squeezenet(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super(squeezenet, self).__init__()
        pretrained_features = tv.squeezenet1_1(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.slice6 = torch.nn.Sequential()
        self.slice7 = torch.nn.Sequential()
        self.N_slices = 7
        for x in range(2):
            self.slice1.add_module(str(x), pretrained_features[x])
        for x in range(2,5):
            self.slice2.add_module(str(x), pretrained_features[x])
        for x in range(5, 8):
            self.slice3.add_module(str(x), pretrained_features[x])
        for x in range(8, 10):
            self.slice4.add_module(str(x), pretrained_features[x])
        for x in range(10, 11):
            self.slice5.add_module(str(x), pretrained_features[x])
        for x in range(11, 12):
            self.slice6.add_module(str(x), pretrained_features[x])
        for x in range(12, 13):
            self.slice7.add_module(str(x), pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1 = h
        h = self.slice2(h)
        h_relu2 = h
        h = self.slice3(h)
        h_relu3 = h
        h = self.slice4(h)
        h_relu4 = h
        h = self.slice5(h)
        h_relu5 = h
        h = self.slice6(h)
        h_relu6 = h
        h = self.slice7(h)
        h_relu7 = h
        vgg_outputs = namedtuple("SqueezeOutputs", ['relu1','relu2','relu3','relu4','relu5','relu6','relu7'])
        out = vgg_outputs(h_relu1,h_relu2,h_relu3,h_relu4,h_relu5,h_relu6,h_relu7)

        return out


class alexnet(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super(alexnet, self).__init__()
        alexnet_pretrained_features = tv.alexnet(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(2):
            self.slice1.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(2, 5):
            self.slice2.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(5, 8):
            self.slice3.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(8, 10):
            self.slice4.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(10, 12):
            self.slice5.add_module(str(x), alexnet_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1 = h
        h = self.slice2(h)
        h_relu2 = h
        h = self.slice3(h)
        h_relu3 = h
        h = self.slice4(h)
        h_relu4 = h
        h = self.slice5(h)
        h_relu5 = h
        alexnet_outputs = namedtuple("AlexnetOutputs", ['relu1', 'relu2', 'relu3', 'relu4', 'relu5'])
        out = alexnet_outputs(h_relu1, h_relu2, h_relu3, h_relu4, h_relu5)

        return out

class vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super(vgg16, self).__init__()
        weights = tv.VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        vgg_pretrained_features = tv.vgg16(weights=weights).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)

        return out

class dinov2(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True, model='dinov2_vits14'):
        super(dinov2, self).__init__()
        dinov2_model = torch.hub.load('facebookresearch/dinov2', model)

        # (patch_embed): PatchEmbed(
        #     (proj): Conv2d(3, 384, kernel_size=(14, 14), stride=(14, 14))
        #     (norm): Identity()
        # )
        # (blocks): ModuleList(
        #     (0-11): 12 x NestedTensorBlock(
        #     (norm1): LayerNorm((384,), eps=1e-06, elementwise_affine=True)
        #     (attn): MemEffAttention(
        #         (qkv): Linear(in_features=384, out_features=1152, bias=True)
        #         (attn_drop): Dropout(p=0.0, inplace=False)
        #         (proj): Linear(in_features=384, out_features=384, bias=True)
        #         (proj_drop): Dropout(p=0.0, inplace=False)
        #     )
        #     (ls1): LayerScale()
        #     (drop_path1): Identity()
        #     (norm2): LayerNorm((384,), eps=1e-06, elementwise_affine=True)
        #     (mlp): Mlp(
        #         (fc1): Linear(in_features=384, out_features=1536, bias=True)
        #         (act): GELU(approximate='none')
        #         (fc2): Linear(in_features=1536, out_features=384, bias=True)
        #         (drop): Dropout(p=0.0, inplace=False)
        #     )
        #     (ls2): LayerScale()
        #     (drop_path2): Identity()
        #     )
        # )
        # (norm): LayerNorm((384,), eps=1e-06, elementwise_affine=True)
        # (head): Identity()

        self.patch_embed = dinov2_model.patch_embed
        self.blocks = dinov2_model.blocks
        self.norm = dinov2_model.norm
        self.head = dinov2_model.head

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def respatialize(self, x: torch.Tensor):
        B, S, C = x.shape
        x = x.permute(0,2,1)
        x = x.reshape((B, C, int(math.sqrt(S)), -1))
        return x

    def forward(self, X):
        h = self.patch_embed(X)
        relus = []
        for idx, block in enumerate(self.blocks):
            # if idx > 0:
            #     relus.append(self.respatialize(h))
            h = block(h)
            relus.append(self.respatialize(h))
        h = self.norm(h)
        # relus.append(self.respatialize(h))
        h = self.head(h)
        dinov2_outputs = namedtuple("DinoV2Outputs", ['relu1', 'relu2', 'relu3', 'relu4', 'relu5', 'relu6', 'relu7', 'relu8', 'relu9', 'relu10', 'relu11', 'relu12'])
        out = dinov2_outputs._make(relus)
        return out

class efficientnetv2(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True, size='s'):
        super(efficientnetv2, self).__init__()
        if(size=='s'):
            effnet2_pretrained_features = tv.efficientnet_v2_s(pretrained=pretrained).features
        elif(size=='m'):
            effnet2_pretrained_features = tv.efficientnet_v2_m(pretrained=pretrained).features
        elif(size=='l'):
            effnet2_pretrained_features = tv.efficientnet_v2_l(pretrained=pretrained).features

        # <class 'torchvision.ops.misc.Conv2dNormActivation'>
        # <class 'torch.nn.modules.container.Sequential'>
        # <class 'torch.nn.modules.container.Sequential'>
        # <class 'torch.nn.modules.container.Sequential'>
        # <class 'torch.nn.modules.container.Sequential'>
        # <class 'torch.nn.modules.container.Sequential'>
        # <class 'torch.nn.modules.container.Sequential'>
        # <class 'torchvision.ops.misc.Conv2dNormActivation'>

        self.norm_in = effnet2_pretrained_features[0]
        self.seq_1 = effnet2_pretrained_features[1]
        self.seq_2 = effnet2_pretrained_features[2]
        self.seq_3 = effnet2_pretrained_features[3]
        self.seq_4 = effnet2_pretrained_features[4]
        self.seq_5 = effnet2_pretrained_features[5]
        self.seq_6 = effnet2_pretrained_features[6]
        self.norm_out = effnet2_pretrained_features[7]
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.norm_in(X)
        h = self.seq_1(h)
        h_relu1 = h
        h = self.seq_2(h)
        h_relu2 = h
        h = self.seq_3(h)
        h_relu3 = h
        h = self.seq_4(h)
        h_relu4 = h
        h = self.seq_5(h)
        h_relu5 = h
        h = self.seq_6(h)
        h_relu6 = h
        self.norm_out(h)
        efficientnetv2_outputs = namedtuple("EffNetV2Outputs", ['relu1', 'relu2', 'relu3', 'relu4', 'relu5', 'relu6'])
        out = efficientnetv2_outputs(h_relu1, h_relu2, h_relu3, h_relu4, h_relu5, h_relu6)
        return out

class resnet(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True, num=18):
        super(resnet, self).__init__()
        if(num==18):
            self.net = tv.resnet18(weights=tv.ResNet18_Weights.IMAGENET1K_V1)
        elif(num==34):
            self.net = tv.resnet34(weights=tv.ResNet34_Weights.IMAGENET1K_V1)
        elif(num==50):
            self.net = tv.resnet50(weights=tv.ResNet50_Weights.IMAGENET1K_V1)
        elif(num==101):
            self.net = tv.resnet101(weights=tv.ResNet101_Weights.IMAGENET1K_V1)
        elif(num==152):
            self.net = tv.resnet152(weights=tv.ResNet152_Weights.IMAGENET1K_V1)
        self.N_slices = 5

        self.conv1 = self.net.conv1
        self.bn1 = self.net.bn1
        self.relu = self.net.relu
        self.maxpool = self.net.maxpool
        self.layer1 = self.net.layer1
        self.layer2 = self.net.layer2
        self.layer3 = self.net.layer3
        self.layer4 = self.net.layer4

    def forward(self, X):
        h = self.conv1(X)
        h = self.bn1(h)
        h = self.relu(h)
        h_relu1 = h
        h = self.maxpool(h)
        h = self.layer1(h)
        h_conv2 = h
        h = self.layer2(h)
        h_conv3 = h
        h = self.layer3(h)
        h_conv4 = h
        h = self.layer4(h)
        h_conv5 = h

        outputs = namedtuple("Outputs", ['relu1','conv2','conv3','conv4','conv5'])
        out = outputs(h_relu1, h_conv2, h_conv3, h_conv4, h_conv5)

        return out

