import torch
import time
import os
from collections import Iterable
import numpy as np
import sys
import cv2
from lrh_utils.utils.cv2_utils import concatImage
import torch.nn.functional as F
import faiss
from retrieval_runner.metric_agent import MetricAgent
from torch.nn import CrossEntropyLoss
from pytorch_loss import LabelSmoothSoftmaxCEV3, AMSoftmax
from torch.cuda.amp import autocast, GradScaler
from tools.lrh_utils.labelsmooth import LabelSmoothingLoss

# CELoss = CrossEntropyLoss()
labelsmooth = LabelSmoothingLoss(116, 0.2)
# LSCELoss = LabelSmoothSoftmaxCEV3(lb_smooth=0.2)
# AMSoftmaxLoss = AMSoftmax(in_feats=1024, n_classes=116).cuda()
metric_agent = MetricAgent()


class Agent(object):
    def __init__(self, model, device_info, save_dir=None, val_fre=1, num_save_image=5):

        if save_dir is None:
            timer = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            save_dir = "./checkpoints/save_{}".format(timer)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.log = open(save_dir + "/Log.log", "w", encoding="utf-8")

        self.save_dir = save_dir
        self.device = device_info["device"]

        # self.saver=None
        self.model = model
        self.ParallelModel = torch.nn.DataParallel(model, device_ids=device_info["device_ids"])
        self.ParallelModel.to(self.device)
        self.Parallel = True

        # 控制模型验证频率
        self.val_fre = val_fre

        # 控制保存图像频率
        self.num_save_image = num_save_image
        self.count_save_image = 0
        self.epoch_start = None
        self.epoch_end = None

        #
        self.pre_save_name = None
        self.pre_best_model_name = None

        self.reduceLR = None
        self.epoch_start = 1

    def summary(self):
        print(self.model)
        self.log.write(str(self.model) + "\n")

    def compile(self, loss_dict, optimizer, metrics):
        self.loss_dict = loss_dict
        self.optimizer = optimizer
        self.metrics = metrics

    def fit_generator(self, train_generator, val_generator, epochs, reduceLR=None, earlyStopping=None,
                      continue_train=False,
                      **kwargs):
        self.reduceLR = reduceLR
        # self.epoch_start = 1
        self.epoch_end = self.epoch_start + epochs - 1
        if continue_train:
            self.continue_train()
        metric = self.metrics
        loss_dict = self.loss_dict
        # 记录某个评价指标用于调整学习率和保存最好模型
        lr_metrics = []

        # 在训练最开始之前实例化一个GradScaler对象
        self.scaler = GradScaler()
        for epoch in range(self.epoch_start, self.epoch_end + 1):
            s = "epoch:{}-lr:{:.8f}".format(epoch, self.optimizer.state_dict()['param_groups'][0]['lr']) + "-" * 5
            print(s)
            self.log.write(s + "\n")
            # train
            phase = "train"
            self.model.train()
            result_epoch = self.iter_on_a_epoch(phase, train_generator, loss_dict, metric, epoch)
            if epoch % self.val_fre != 0:
                continue
            # valid
            phase = "valid"
            metric.reset()
            self.model.eval()
            with torch.no_grad():
                result_epoch = self.iter_on_a_epoch(phase, val_generator, loss_dict, metric, epoch)
                # 使用"miou_metrics"作为lr_metrics
                lr_metrics.append(result_epoch["ACC" + "_metrics"])

            # 保存模型
            # save cur
            save_name = "epo_{}-score_{:.5f}.pth".format(epoch, lr_metrics[-1])
            self.save_model(save_name)
            if self.pre_save_name is not None:
                os.remove(self.save_dir + "/model/" + self.pre_save_name)
            self.pre_save_name = save_name
            # save best
            if lr_metrics[-1] == np.max(lr_metrics):
                best_model_name = "best_epo_{}-score_{:.5f}.pth".format(epoch, lr_metrics[-1])
                self.save_model(best_model_name)
                if self.pre_best_model_name is not None:
                    os.remove(self.save_dir + "/model/" + self.pre_best_model_name)
                self.pre_best_model_name = best_model_name

            # recude lr
            if self.reduceLR is not None:
                epoch_loss = sum([val for key, val in result_epoch.items() if "loss" in key])
                if isinstance(self.reduceLR, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.reduceLR.step(lr_metrics[-1], epoch)
                if isinstance(self.reduceLR, torch.optim.lr_scheduler.StepLR) or \
                        isinstance(self.reduceLR, torch.optim.lr_scheduler.MultiStepLR):
                    self.reduceLR.step()
            # earlyStopping

            if earlyStopping is not None:
                earlyStopping.step(lr_metrics[-1])
                if earlyStopping.early_stop:
                    break

        # # load_best and save image
        # print("load best model and save image")
        # if self.pre_best_model_name is None:
        #     raise
        #
        # # save valid
        # self.load_weights(self.save_dir + "/model/" + self.pre_best_model_name)
        # metric.reset()
        # self.model.eval()
        # with torch.no_grad():
        #     self.num_save_image = 1000000
        #     self.iter_on_a_epoch("valid", val_generator, loss_dict, metric, epoch="best")

    def iter_on_a_epoch(self, phase, dataloader, loss_dict, metric, epoch, **kwargs):
        assert phase in ["train", "valid", "test"]

        result_epoch = {"count": 0}
        metric.reset()
        time_epoch = 0
        if phase == "valid":
            gallery_list = []
            gallery_label_list = []
            model = self.ParallelModel if self.Parallel else self.model
            print("getting gallery")
            for cnt_batch, batch in enumerate(dataloader["key"]):
                img_tensor, label_tensor = batch
                # 用于训练的转为tensor
                img_tensor = img_tensor.to(self.device)
                cls_score, feat, global_feat = model(img_tensor)
                gallery_list.append(feat.view(feat.shape[0], -1))
                gallery_label_list.append(label_tensor)
            gallery = torch.cat(gallery_list, dim=0).detach().cpu().numpy()
            gallery_label = torch.cat(gallery_label_list, dim=0).detach().cpu().numpy()
            metric_agent.set_gallery(gallery=gallery, gallery_label=gallery_label)
            print("getting gallery done!")
        for cnt_batch, batch in enumerate(dataloader["quiry"]):
            time_start = time.time()
            result = self.iter_on_a_batch(batch, loss_dict=loss_dict, phase=phase, epoch=epoch)
            if self.model.training and self.reduceLR is not None and isinstance(self.reduceLR,
                                                                                torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
                self.reduceLR.step(epoch - 1 + cnt_batch / len(dataloader["quiry"]))
            # 返回结果
            pred_batch, label_batch = result["batch"]
            # 结果评估
            # metric.add_batch(pixel_score_batch, pixel_label_scale_tensor)
            metric.add_batch(label_batch, pred_batch)
            # log of every batch
            time_end = time.time()
            time_batch = (time_end - time_start)
            time_epoch += time_batch
            time_lack = time_batch * (len(dataloader["quiry"]) + 1 - cnt_batch)
            s = "{}: batch{}/{} [{}<{}, {:.2f}s/it]    ".format(phase, cnt_batch + 1, len(dataloader["quiry"]),
                                                                Agent.t2ms(time_epoch),
                                                                Agent.t2ms(time_epoch + time_lack), time_batch)

            # 返回损失
            if not self.model.training:
                sys.stdout.write("\r" + s)
                sys.stdout.flush()
                continue

            result_epoch["count"] += label_batch.shape[0]
            for key, val in result["loss"].items():
                key = key + "_loss"
                if key not in result_epoch.keys(): result_epoch[key] = []
                result_epoch[key].append(val)
                s += "---{}: {:.4f}    ".format(key, val)

            sys.stdout.write("\r" + s)
            sys.stdout.flush()
        # log of every epoch
        s = "{}: batch{}/{} [{}<{}, {:.2f}s/it]    ".format(phase, len(dataloader["quiry"]), len(dataloader["quiry"]),
                                                            Agent.t2ms(time_epoch), Agent.t2ms(time_epoch),
                                                            time_epoch / len(dataloader["quiry"]))
        # 将所有loss平均
        for key, val in result_epoch.items():
            if "loss" in key:
                result_epoch[key] = np.array(val).sum() / len(val)

                s += "---{}: {:.4f}    ".format(key, result_epoch[key])
        metric_dict = metric.apply()
        for key, val in metric_dict.items():
            key = key + "_metrics"
            result_epoch[key] = val
            s += "---{}: {:.4f}    ".format(key, result_epoch[key])

        print("\r" + s)
        self.log.write(s + "\n")

        self.count_save_image = 0
        metric_agent.reset_gallery()
        return result_epoch

    def iter_on_a_batch(self, batch, phase, loss_dict, epoch):
        """
        :param batch: img_batch [np.float32, (b, c, h, w)], label_batch [np.int32, (b, c, h, w)], pixel_label_batch [np.int32, (b, )]
        :param phase:
        :param loss_dict:
        :param epoch:
        :return:
        """

        # only trans to tensor when going forward.
        assert phase in ["train", "valid", "test"], print(phase)
        result = dict()
        img_tensor, label_tensor = batch
        model = self.ParallelModel if self.Parallel else self.model
        # model = self.model
        optimizer = self.optimizer
        device = self.device

        # # 用于训练的转为tensor
        # lam = np.random.beta(2, 2)
        # index = torch.randperm(img_tensor.size(0)).cuda()
        # img_tensor = lam*img_tensor + (1-lam)*img_tensor[index, :]
        #
        # exit()
        img_tensor = img_tensor.to(device)
        label_tensor = label_tensor.to(device)
        # 前向过程(model + loss)开启 autocast
        with autocast():
            if phase == "train":
                cls_score, feat, global_feat = model(img_tensor)
            else:
                cls_score, feat, global_feat = model(img_tensor)

            ###### cul loss
            losses = dict()
            # if phase in ["train", "valid"]:
            if phase == "train":
                # for name, loss_fun in loss_dict.items():
                #     loss = loss_fun(score_tensor, label_tensor)
                #     losses[name] = loss
                # losses["CE"] = CELoss(cls_score, label_tensor)
                losses["LSCE"] = labelsmooth(cls_score, label_tensor)
                # losses["AMSoftmax"] = AMSoftmaxLoss(feat, label_tensor)
                loss_sum = sum(list(losses.values()))
                losses["loss_sum"] = float(loss_sum.item())
                result["loss"] = losses

        ##### backward when training
        if phase == "train":
            model.zero_grad()
            # 单精度
            # loss_sum.backward()
            # optimizer.step()

            # 半精度
            # Scales loss. 为了梯度放大.
            self.scaler.scale(loss_sum).backward()

            # scaler.step() 首先把梯度的值unscale回来.
            # 如果梯度的值不是 infs 或者 NaNs, 那么调用optimizer.step()来更新权重,
            # 否则，忽略step调用，从而保证权重不更新（不被破坏）
            self.scaler.step(optimizer)

            # 准备着，看是否要增大scaler
            self.scaler.update()

        #### return
        if phase == "train":
            pred_batch = torch.argmax(cls_score, dim=-1).detach().cpu().numpy().astype(np.int32)
            for k, v in result["loss"].items():
                if k != "loss_sum":
                    result["loss"][k] = result["loss"][k].detach().cpu()
        else:
            pred_batch = metric_agent.run_metric(feat.detach().cpu().numpy())
        label_batch = label_tensor.detach().cpu().numpy().astype(np.int32)

        result["batch"] = pred_batch, label_batch
        return result

    def continue_train(self):
        model_dir = self.save_dir + "/model/"
        if not os.path.exists(model_dir):
            print("connot find file:{}, training from the beginning.".format(model_dir))
            return
        name_list = os.listdir(model_dir)
        if len(name_list) == 0:
            print("cannot find any model to continue training, training from the beginning.")
            return
        epoch_list = [int(name.split("-")[0].split("_")[-1]) for name in name_list]
        index = np.argmax(epoch_list)
        epoch = epoch_list[index]
        model_name = name_list[index]
        print("continue training by {}".format(model_name))
        self.load_weights(model_dir + model_name)
        self.epoch_start = epoch + 1

    def load_weights(self, load_path):
        if os.path.exists(load_path):
            pthfile = torch.load(load_path)
            # print(pthfile.keys())
            self.model.load_state_dict(pthfile, strict=False)
            print("load weights from {}".format(load_path))
        else:
            raise Exception("Load model falied, {} is not existing!!!".format(load_path))

    def save_model(self, save_name):
        save_dir = self.save_dir + "/model/"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, save_name)
        print("save weights to {}".format(save_path))
        torch.save(self.model.state_dict(), save_path)

    def load_best_model(self):

        load_names = [name for name in os.listdir(self.save_dir + "/model/") if name.endswith(".pth")]
        load_name = sorted(load_names, key=lambda x: float(x.split(".")[-2]), reverse=True)[0]

    @staticmethod
    def t2ms(t):
        m = int(t // 60)
        if m > 99:
            return "99:59"
        s = int(t - m * 60)
        return "{:0>2d}:{:0>2d}".format(m, s)
