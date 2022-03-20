from torch import nn
import torch
import torch.nn.functional as F


class Mish(nn.Module):
    def __init__(self):
        super().__init__()
        print("Mish activation loaded...")
    def forward(self,x):
        x = x * (torch.tanh(F.softplus(x)))
        return x


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    ac = Mish()
    x = np.linspace(-5, 5, 1000, dtype=np.float32)
    y = (ac(torch.tensor(x, dtype=torch.float32))).numpy().astype(np.float32)
    plt.plot(x, y)
    plt.show()
    # print(x)
    # print(y)