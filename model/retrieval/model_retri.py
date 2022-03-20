from model.retrieval import resnet_copy, senet_copy, efficientnet_copy, swin_transformer_copy, volo_copy, cswin_transformer_copy
from torch import nn
import torch
from model.retrieval.outlook_attention import OutlookAttention
from model.retrieval.ScaledDotProductAttention_copy import ScaledDotProductAttention
# from facenet_pytorch import MTCNN
import torch.nn.functional as F
from tools.pyretri.extract.aggregator import GeM, GAP, GMP, Crow, SCDA, SPoC, RMAC, PWA
import random



def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

class AMSoftmax(nn.Module):

    def __init__(self,
                 in_feats,
                 n_classes,
                 m=0.3,
                 s=15):
        super(AMSoftmax, self).__init__()
        self.m = m
        self.s = s
        self.in_feats = in_feats
        self.W = torch.nn.Parameter(torch.randn(in_feats, n_classes), requires_grad=True)
        nn.init.xavier_normal_(self.W, gain=1)

    def forward(self, x, lb):
        assert x.size()[0] == lb.size()[0]
        assert x.size()[1] == self.in_feats
        x_norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-9)
        x_norm = torch.div(x, x_norm)
        w_norm = torch.norm(self.W, p=2, dim=0, keepdim=True).clamp(min=1e-9)
        w_norm = torch.div(self.W, w_norm)
        costh = torch.mm(x_norm, w_norm)
        delt_costh = torch.zeros_like(costh).scatter_(1, lb.unsqueeze(1), self.m)
        costh_m = costh - delt_costh
        costh_m_s = self.s * costh_m
        return costh_m_s
        # loss = self.ce(costh_m_s, lb)
        # return loss

class NormLayer(nn.Module):
    def __init__(self, p=2):
        super(NormLayer, self).__init__()

        self.p = p

    def forward(self, x):
        x_norm = torch.norm(x, p=self.p, dim=1, keepdim=True).clamp(min=1e-9)
        x_norm = torch.div(x, x_norm)
        return x_norm


class Model(nn.Module):
    in_planes = 2048

    def __init__(self, model_name, num_classes, neck, model_path=None, no_feat=False):
        super(Model, self).__init__()
        feature_dim = 2048
        device = torch.device("cuda")
        self.centers = (torch.rand(num_classes, feature_dim).to(device) - 0.5) * 2
        self.no_feat = no_feat
        if ("resnet" in model_name or "resnext" in model_name) and "se" not in model_name:
            if model_path is not None:
                self.base = getattr(resnet_copy, model_name)(pretrained=False)
                self.base.load_state_dict(torch.load(model_path))
            else:
                self.base = getattr(resnet_copy, model_name)(pretrained=True)
            self.in_planes = self.base.fc.in_features
            del self.base.fc
        elif "se" in model_name:
            if model_path is not None:
                self.base = getattr(senet_copy, model_name)(pretrained=False)
                self.base.load_state_dict(torch.load(model_path))
            else:
                self.base = getattr(senet_copy, model_name)(pretrained=True)
            self.in_planes = 2048
            del self.base.dropout
        elif "efficientnet" in model_name:
            self.base = efficientnet_copy.EfficientNet.from_pretrained(model_name, model_path)
            self.in_planes = self.base._fc.in_features
            del self.base._fc
        elif "cswin" in model_name:
           cswin_name = {
               "cswin_t224": "CSWin_64_12211_tiny_224",
               "cswin_s224": "CSWin_64_24322_small_224",
               "cswin_b224": "CSWin_96_24322_base_224",
               "cswin_l224": "CSWin_144_24322_large_224",
               "cswin_b384": "CSWin_96_24322_base_384",
               "cswin_l384": "CSWin_144_24322_large_384",
           }
           self.base = getattr(cswin_transformer_copy, cswin_name[model_name])(pretrained=True)
           self.in_planes = self.base.head.in_features
           del self.base.head
        elif "swin" in model_name:
            self.base = getattr(swin_transformer_copy, model_name)(pretrained=True)
            self.in_planes = self.base.head.in_features
            del self.base.head
        elif "volo" in model_name:
            self.base = getattr(volo_copy, model_name)(pretrained=True)
            self.in_planes = self.base.head.in_features
            del self.base.head
            if self.base.return_dense:
                del self.base.aux_head
        # self.gap = nn.AdaptiveAvgPool1d(1) if "swin" in model_name else nn.AdaptiveAvgPool2d(1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes
        self.neck = neck
        # self.squeeze_layer = nn.Conv2d(self.in_planes, 1024, 1, 1, bias=False)

        self.squeeze_layer = None
        self.attention_layer = OutlookAttention(dim=self.in_planes)
        # self.attention_layer = ScaledDotProductAttention(d_model=self.in_planes, d_k=self.in_planes, d_v=self.in_planes,
        #                                                  h=8)
        # self.attention_layer = None


        if self.neck == 'no':
            # self.classifier = nn.Linear(self.in_planes, self.num_classes)
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)  # new add by luo
            self.classifier.apply(weights_init_classifier)  # new add by luo
        elif self.neck == 'bnneck':
            self.bottleneck = nn.BatchNorm1d(self.in_planes)
            self.bottleneck.bias.requires_grad_(False)  # no shift
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)

            self.bottleneck.apply(weights_init_kaiming)
            self.classifier.apply(weights_init_classifier)

        self.norm_layer = None
        # self.classifier = AMSoftmax(self.in_planes, n_classes=num_classes)


        agg_list = [GAP()]
        # agg_list = [G]
        # if ~, need relu
        # not useful: SCDA, RMAC
        # useful: Crow~, GeM~, SPoC
        # wait: PWA
        assert len(agg_list) > 0
        self.aggregator = lambda x: torch.cat([agg({"fea": x})["fea_" + agg.__class__.__name__] for agg in agg_list], dim=-1)
        print("using aggregator as follow:")
        print([a.__class__.__name__ for a in agg_list])


        # just for one agg_list

        # TODO add drop out
        # self.dropout = nn.Dropout(0.5)


    # def forward(self, x):
    #
    #     x = self.base(x)
    #     x = self.squeeze_layer(x) if self.squeeze_layer is not None else x
    #     x = self.attention_layer(x) if self.attention_layer is not None else x
    #     global_feat = self.gap(x)  # (b, 2048, 1, 1)
    #     global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)
    #
    #     if self.neck == 'bnneck':
    #         feat = self.bottleneck(global_feat)  # normalize for angular softmax
    #     else:
    #         feat = global_feat
    #
    #     # cls_score = self.classifier(feat)
    #     # TODO
    #     # feat = self.dropout(feat)
    #     if self.training:
    #         cls_score = self.classifier(feat)
    #     else:
    #         cls_score = None
    #     return cls_score, feat, global_feat  # global feature for triplet loss

    def forward(self, x, y=None):
        if self.training:
            # not useful
            # b, c, h, w = x.shape
            # p = random.random() * 0.2
            # x[:b//2] = x[:b//2] * (1 - p) + x[b//2:] * p
            x = self.base(x)
            x = self.squeeze_layer(x) if self.squeeze_layer is not None else x
            x = self.attention_layer(x) if self.attention_layer is not None else x
            global_feat = self.gap(x)  # (b, 2048, 1, 1)
            global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)


            if self.neck == 'bnneck':
                feat = self.bottleneck(global_feat)  # normalize for angular softmax
            else:
                feat = global_feat
            # feat = self.dropout(feat)
            cls_score = self.classifier(feat)
            return cls_score, feat, global_feat  # global feature for triplet loss
        else:
            # xx
            # b, c, h, w = x.shape
            # x1, x2 = x[..., :w//2], x[..., w//2:]
            # x = x1
            #
            x1 = F.adaptive_avg_pool2d(x, (128, 128))
            x = self.base(x)
            x = self.squeeze_layer(x) if self.squeeze_layer is not None else x
            x = self.attention_layer(x) if self.attention_layer is not None else x
            x = F.relu(x, inplace=True)
            # global_feat = self.aggregator(x)
            global_feat = self.gap(x)  # (b, 2048, 1, 1)
            global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)
            # global_feat = self.norm_layer(global_feat) if self.norm_layer is not None else global_feat


            # multi-scale
            x1 = self.base(x1)
            x1 = self.squeeze_layer(x1) if self.squeeze_layer is not None else x
            x1 = self.attention_layer(x1) if self.attention_layer is not None else x
            x1 = F.relu(x1, inplace=True)
            global_feat1 = self.gap(x1)
            global_feat1 = global_feat1.view(global_feat1.shape[0], -1)
            # global_feat1 = self.norm_layer(global_feat1) if self.norm_layer is not None else global_feat1

            global_feat = torch.cat([global_feat, global_feat1], dim=-1)





#####################
            # x = x2
            # x = self.base(x)
            # x = self.squeeze_layer(x) if self.squeeze_layer is not None else x
            # x = self.attention_layer(x) if self.attention_layer is not None else x
            #
            # # 加个relu
            # x = F.relu(x, inplace=True)
            # # 尝试其他aggregation
            # # global_feat = self.aggregator(x)
            # global_feat2 = self.gap(x)  # (b, 2048, 1, 1)
            # global_feat2 = global_feat2.view(global_feat2.shape[0], -1)  # flatten to (bs, 2048)
            #
            # # 测试使用norm
            # global_feat2 = self.norm_layer(global_feat2) if self.norm_layer is not None else global_feat2
            # global_feat = torch.cat([global_feat, global_feat2], dim=-1)
############
            # 增加多尺度输入提取特征
            # useful: 0.95635->0.95651，提交下降了,
            # x1 = F.adaptive_avg_pool2d(x, (128, 128))
            # x1 = self.squeeze_layer(x1) if self.squeeze_layer is not None else x
            # x1 = self.attention_layer(x1) if self.attention_layer is not None else x
            # global_feat1 = self.gap(x1)  # (b, 2048, 1, 1)
            # global_feat1 = global_feat1.view(global_feat1.shape[0], -1)  # flatten to (bs, 2048)
            # global_feat = torch.cat([global_feat, global_feat1], dim=-1)

            if self.neck == 'bnneck':
                feat = self.bottleneck(global_feat)  # normalize for angular softmax
            else:
                feat = global_feat
            # TODO
            cls_score = None
            return cls_score, feat, global_feat  # global feature for triplet loss




if __name__ == "__main__":
    # model = Model(model_name="resnext50_32x4d", num_classes=116, neck="bnneck")
    # model = Model(model_name="resnet18", num_classes=116, neck="bnneck")
    model = Model(model_name="volo_d5", num_classes=116, neck="no")
    for cnt, (name, module) in enumerate(model.base.named_children()):
        print(cnt, name)

    x = torch.rand(2, 3, 224, 224).cuda()
    model.train()
    model = model.cuda()
    cls_score, feat, global_feat = model(x)
    print(cls_score.size)
    print(feat.shape)
    print(global_feat.shape)
    # model.eval()
    # _, feat, global_feat = model(x)



