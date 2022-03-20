from torch.nn import functional as F
import torch
# from torchcam.cams import C
"""
    torchcam
"""
class CAM(object):
    def __init__(self, model, weight_layer, cam_layer, uppool="nearest"):
        self.model = model
        self.handles = []
        self.__init_hook(cam_layer)
        self.weight_layer = weight_layer
        self.features_map = None
        self.layer_grad = None
        self.uppool = uppool
        assert uppool is None or uppool in ["nearest", "linear", "bilinear", "bicubic", "trilinear"]


    def __init_hook(self, cam_layer):

        for name, module in self.model.named_modules():
            if name == cam_layer:
                self.handles.append(module.register_forward_hook(self.hook_fn_forward))
        assert len(self.handles) == 1

    def __call__(self, input, label):
        """

        :param input: torch.tensor [C, H, W]
        :param label: torch.tensor or int, only accept scalar.
        :return: torch.tensor [1, H, W]
        """
        y = self.model(input.unsqueeze(0))
        cam = self.get_cam(label)
        if self.uppool is not None:
            cam = torch.nn.Upsample(input.shape, mode=self.uppool)
        return cam, torch.argmax(y[0])

    def get_cam(self, label):
        weight = None
        for name, param in self.model.named_parameters():
            if name == self.weight_layer:
                weight = param[label]
                break
        assert weight is not None
        cam = torch.mean(torch.mul(self.features_map, weight.unsqueeze(-1).unsqueeze(-1)), dim=0)
        cam = F.relu(cam)
        # 归一化
        Max = torch.max(cam)
        Min = torch.min(cam)
        cam = (cam - Min) / (Max - Min)
        self.features_map = None
        self.layer_grad = None
        return cam


    def hook_fn_forward(self, module, input, output):
        assert len(output) == 1 and self.features_map is None
        self.features_map = output[0]


class GradCAM(object):
    def __init__(self, model, cam_layer, uppool="nearest"):
        self.model = model
        self.handles = []
        self.__init_hook(cam_layer)
        self.features_map = None
        self.layer_grad = None
        self.uppool = uppool
        assert uppool is None or uppool in ["nearest", "linear", "bilinear", "bicubic", "trilinear"]


    def __init_hook(self, cam_layer):

        for name, module in self.model.named_modules():
            if name == cam_layer:
                self.handles.append(module.register_forward_hook(self.hook_fn_forward))
                self.handles.append(module.register_backward_hook(self.hook_fn_back))
        assert len(self.handles) == 2

    def __call__(self, input, label):
        """

        :param input: torch.tensor [C, H, W]
        :param label: torch.tensor or int, only accept scalar.
        :return: torch.tensor [1, H, W]
        """
        y = self.model(input.unsqueeze(0))
        y[0][label].backward()
        cam = self.get_cam()
        if self.uppool is not None:
            cam = torch.nn.Upsample(size=input.shape[1:], mode=self.uppool)(cam[None][None])
            cam = cam[0][0]
        return cam, torch.argmax(y[0])

    # def get_cam(self):
    #     weight = F.adaptive_avg_pool2d(self.layer_grad, (1, 1))
    #     cam = torch.mean(torch.mul(self.layer_grad, weight), dim=1)
    #     cam = F.relu(cam)
    #     # 归一化
    #     Max = torch.max(cam)
    #     Min = torch.min(cam)
    #     cam = (cam - Min) / (Max - Min)
    #     self.layer_grad = None
    #     return cam

    def get_cam(self):
        weight = torch.mean(self.layer_grad, dim=[-1, -2], keepdim=True)
        cam = torch.mean(torch.mul(self.features_map, weight), dim=0)
        cam = F.relu(cam)
        # 归一化
        Max = torch.max(cam)
        Min = torch.min(cam)
        cam = (cam - Min) / (Max - Min)
        self.features_map = None
        self.layer_grad = None
        return cam


    def hook_fn_forward(self, module, input, output):
        assert len(output) == 1 and self.features_map is None
        self.features_map = output[0]

    def hook_fn_back(self, module, input, output):
        assert len(output) == 1 and self.layer_grad is None
        self.layer_grad = output[0][0]





if __name__ == "__main__":
    from torchvision.models import resnet18
    import torch

    grad_cam = GradCAM(model=resnet18(pretrained=False), cam_layer="layer4.1.bn2", uppool="nearest")
    cam, _ = grad_cam(torch.rand((3, 128, 128)).requires_grad_(), 0)
    print(cam.shape)

    # n_cam = CAM(model=resnet18(pretrained=False), weight_layer="fc.weight", cam_layer="layer4.1.bn2", uppool=None)
    # cam = n_cam(torch.rand((3, 128, 128)).requires_grad_(), 0)