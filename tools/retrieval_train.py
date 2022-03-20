import os
import sys

from lrh_utils.pytorch_tools import setup_seed

# set random seed
seed = 2021
setup_seed(seed)

import torch
from torch.optim import SGD, lr_scheduler, Adam, AdamW
from torch.utils.data import DataLoader
from lrh_utils.pytorch_tools.sampler import EnlargeLabelShufflingSampler, UnderShuffingSampler
from lrh_utils.pytorch_tools.earlystop import EarlyStopping
from retrieval_runner.torch_func import Agent
from retrieval_runner.dataset import QuiryDataset, KeyDataset
from retrieval_runner.metrics import Metrics
from torch import nn
from random import shuffle

sys.path = [os.path.abspath(os.path.dirname(__file__) + "/../")] + sys.path
from model import Model


def train(**kwargs):
    param = kwargs
    DEVICE_INFO = dict(
        gpu_num=torch.cuda.device_count(),
        device_ids=range(0, torch.cuda.device_count(), 1),
        # device_ids=[0],
        device=torch.device("cuda:0") if torch.cuda.is_available() else "cpu",
        index_cuda=0)

    # param["save_dir"] = param["save_dir"] + "/{}_pretrained_{}_ds{}".format(param["model_name"], param["model_pretrained"], param["label_downsample"])

    # updata from args

    train_data = {
        "quiry": QuiryDataset(root="../data/train/a_images_crop", phase="train", width=param["width"],
                              height=param["height"]),
        "key": KeyDataset(root="../data/train/b_images_crop", width=param["width"], height=param["height"])
    }

    val_data = {
        "quiry": QuiryDataset(root="../data/train/a_images_crop", phase="val", width=param["width"],
                              height=param["height"]),
        "key": KeyDataset(root="../data/train/b_images_crop", width=param["width"], height=param["height"],
                          use_augs=True)
    }

    train_data["quiry"].samples += [("/../b_images_crop" + name, label) for name, label in
                                    train_data["key"].samples]

    train_generator = {
        # todo
        "quiry": DataLoader(dataset=train_data["quiry"], batch_size=param["batch_size"], shuffle=True, num_workers=0,
                            collate_fn=train_data["quiry"].collate_fn),
        # "quiry": DataLoader(dataset=train_data["quiry"], batch_size=param["batch_size"], num_workers=0,
        #                     collate_fn=train_data["quiry"].collate_fn,
        #                     sampler=EnlargeLabelShufflingSampler([sample[1] for sample in train_data["quiry"].samples],
        #                                                          num_limit=len(train_data["quiry"].samples))),

        "key": DataLoader(dataset=train_data["key"], batch_size=param["batch_size"], shuffle=False, num_workers=0,
                          collate_fn=train_data["key"].collate_fn)
    }
    val_generator = {
        "quiry": DataLoader(dataset=val_data["quiry"], batch_size=param["batch_size"] // 4, shuffle=False,
                            num_workers=8,
                            collate_fn=val_data["quiry"].collate_fn),
        "key": DataLoader(dataset=val_data["key"], batch_size=param["batch_size"] // 4, shuffle=False, num_workers=8,
                          collate_fn=val_data["key"].collate_fn)
    }

    train_data["quiry"].samples = train_data["quiry"].samples[:20]

    # model

    # model = Model(model_name=param["model_name"], num_classes=116, neck="bnneck")
    model = Model(model_name=param["model_name"], num_classes=116, neck="no")
    model.load_state_dict(torch.load("/media/gdut502/139e9283-5fa3-498b-ba3d-0272a299eeeb/wqr/xinye/xinye1/model_files/retrieval/efficientnet-b5/model/best_epo_12-score_0.95774.pth"))

    # metrics
    metrics = Metrics(num_classes=116)

    # loss
    # loss_dict = {"CELoss": nn.CrossEntropyLoss()}
    loss_dict = {}

    early_stopping = EarlyStopping(patience=30, mode="max", verbose=True)

    optimizer = None
    if param["optimizer_type"] == "Adam":
        optimizer = Adam(params=model.parameters(), lr=param["lr"], weight_decay=0.0001)
    elif param["optimizer_type"] == "SGD":
        optimizer = SGD(params=model.parameters(), lr=param["lr"], weight_decay=0.0001)
    elif param["optimizer_type"] == "AdamW":
        optimizer = AdamW(params=model.parameters(), lr=param["lr"], weight_decay=0.05)

    # reduceLR = lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=8, verbose=True)
    # reduceLR = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=1, verbose=True)

    # reduceLR = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    # reduceLR = lr_scheduler.MultiStepLR(optimizer, milestones=[4, 8, 10], gamma=0.46)
    reduceLR = lr_scheduler.MultiStepLR(optimizer, milestones=[5, 8, 10, 11], gamma=0.55)

    # TODO
    # reduceLR = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # reduceLR = None

    # get agent
    agent = Agent(model=model, device_info=DEVICE_INFO, save_dir=param["save_dir"], val_fre=param["val_fre"],
                  num_save_image=param["num_save_image"])
    agent.Parallel = True

    if param["resume_from"] is not None:
        model.load_state_dict(torch.load(param["resume_from"]))
        agent.epoch_start = int(param["resume_from"].split("epo_")[-1].split("-")[0]) + 1
        print("model resume from {} in epoch {}".format(param["resume_from"], agent.epoch_start))

    agent.compile(loss_dict=loss_dict, optimizer=optimizer, metrics=metrics)
    agent.summary()

    # train

    agent.fit_generator(train_generator=train_generator,
                        val_generator=val_generator,
                        epochs=param["epochs"],
                        reduceLR=reduceLR,
                        earlyStopping=early_stopping)


if __name__ == "__main__":

    opt = {
        "save_dir": None,
        "val_fre": 1,
        "epochs": 1,
        "comtinue_train": True,
        "num_save_image": 0,
        # "model_name": "resnet18",
        # "model_name": "resnext50_32x4d",
        "model_name": "efficientnet-b5",
        # "model_name": "efficientnet-b6",
        # "model_name": "swin_t224",
        # "model_name": "swin_b224",
        # "model_name": "swin_b384",
        # "model_name": "swin_s224",
        # "model_name": "volo_d3",
        # "model_name": "cswin_b224",

        # data
        "width": 224,
        "height": 224,
        # learning strategy
        "optimizer_type": "AdamW",
        "lr": 0.000,
        "batch_size": 52,
        "resume_from": None,
    }
    if opt["save_dir"] is None:
        opt["save_dir"] = "../model_files/retrieval/{}".format(opt["model_name"])

    train(**opt)
