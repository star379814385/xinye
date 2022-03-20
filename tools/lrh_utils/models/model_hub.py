import torch

"""
    SENet
    https://github.com/moskomule/senet.pytorch
    repo_or_dir = "moskomule/senet.pytorch"
    __all__ = []
     
"""

if __name__ == "__main__":
    # print(torch.hub.list('moskomule/senet.pytorch'))
    model = torch.hub.load('moskomule/senet.pytorch', 'se_resnet20', pretrained=True)
    print(model)