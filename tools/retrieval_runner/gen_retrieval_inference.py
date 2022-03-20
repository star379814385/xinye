import json
import torch
from tools.retrieval_train import DataLoader, QuiryDataset, KeyDataset, Model
from tools.retrieval_runner.metric_agent import MetricAgent
from tqdm import tqdm
import numpy as np
import glob
from tools.pyretri.extract.aggregator import *
from tools.pyretri.index.dim_processor import PCA, L2Normalize
from tools.pyretri.index.feature_enhancer import DBA
from torch.nn import functional as F
import os
import cv2
from random import shuffle

############### set data
roi_size = (224, 224)
query_data = QuiryDataset(root="../../data/初赛数据集/test/a_images_crop", phase="val", width=roi_size[0], height=roi_size[1])

############# use_augs in gallery
gallery_aug = True
gallery_data = KeyDataset(root="../../data/初赛数据集/test/b_images_crop", width=roi_size[0], height=roi_size[1],
                          use_augs=gallery_aug)
query_generator = DataLoader(dataset=query_data, batch_size=64, shuffle=False, num_workers=8,
                             collate_fn=query_data.collate_fn)
gallery_generator = DataLoader(dataset=gallery_data, batch_size=64, shuffle=False, num_workers=16,
                               collate_fn=gallery_data.collate_fn)

#################### load model and load weight
# model_name = "se_resnet50"
# model_name = "resnext50_32x4d"
# model_name = "efficientnet-b5"
model_name = "swin_b224"
# model = Model(model_name=model_name, num_classes=116, neck="bnneck").cuda()
model = Model(model_name=model_name, num_classes=116, neck="no").cuda()
# model_cls = None
weight_path = None
if weight_path is None:
    weight_path = glob.glob("./MetricLearning/checkpoints2/{}/model/best*.pth".format(model_name))
    assert len(weight_path) == 1
    weight_path = weight_path[0]
model.load_state_dict(torch.load(weight_path))
print("model load weight from " + weight_path)

########### just use backbone for extring features
model = model.base
model = model.eval()
##########  set parallel
Parallel = True
model = torch.nn.DataParallel(model, device_ids=range(0, torch.cuda.device_count(), 1)) if Parallel else model
device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
# device = "cpu"

############### change gap to other aggregator
agg_list = [GAP()]
assert len(agg_list) > 0
aggregator = lambda x: torch.cat([agg({"fea": x})["fea_" + agg.__class__.__name__] for agg in agg_list], dim=-1)
print("using aggregator as follow:")
print([a.__class__.__name__ for a in agg_list])

############## save train_data features for dim process
train_data = None
rewrite_train_data_features = False
if rewrite_train_data_features:
    train_data_list = []
    # 使用KeyDataset接口，可以使用图像增强
    train_data_aug = False
    train_query_data = KeyDataset(root="./data/初赛数据集/train/b_images_crop", width=roi_size[0], height=roi_size[1],
                                  use_augs=train_data_aug)
    train_gallery_data = KeyDataset(root="./data/初赛数据集/train/b_images_crop", width=roi_size[0], height=roi_size[1])
    # 训练时使用了query和gallery
    train_query_data.samples += [("/../../train/b_images_crop" + name, label) for name, label in
                                 train_gallery_data.samples]
    with torch.no_grad():
        for batch in tqdm(gallery_generator):
            img_tensor, label_tensor = batch
            # 用于训练的转为tensor
            img_tensor = img_tensor.to(device)
            # forward
            feat = model(img_tensor)
            feat = aggregator(feat)
            # _, feat, _ = model(img_tensor)
            # feat, _, _ = model(img_tensor)
            # feat = F.softmax(feat, dim=-1)
            # to numpy
            train_data_list.append(feat.detach().cpu().numpy())
    train_data = np.concatenate(train_data_list, axis=0)
    np.save("train_data.npy", train_data)
else:
    train_data = np.load("train_data.npy")

############### load gallery features ###########
print("getting gallery and gallery_label from infering key_data.")
rewrite_test_gallery_features = False
gallery = None
gallery_label = None
if rewrite_test_gallery_features:
    gallery_list = []
    gallery_label_list = []
    # gallery_data.samples = gallery_data.samples[:24]
    with torch.no_grad():
        for batch in tqdm(gallery_generator):
            img_tensor, label_tensor = batch
            # 用于训练的转为tensor
            img_tensor = img_tensor.to(device)
            # forward
            feat = model(img_tensor)
            feat = aggregator(feat)
            # _, feat, _ = model(img_tensor)
            # feat, _, _ = model(img_tensor)
            # feat = F.softmax(feat, dim=-1)
            # to numpy
            gallery_list.append(feat.detach().cpu().numpy())
            gallery_label_list.append(label_tensor.detach().cpu().numpy())
    gallery = np.concatenate(gallery_list, axis=0)
    gallery_label = np.concatenate(gallery_label_list, axis=0)
    np.save("test_gallery.npy", gallery)
    np.save("test_gallery_label.npy", gallery_label)
else:
    gallery = np.load("test_gallery.npy")
    gallery_label = np.load("test_gallery_label.npy")

########### dim process
pca_hps = {
    "proj_dim": 512,
    "whiten": False,
    "l2": False,
}
# dim_process = [L2Normalize(), PCA(train_fea=train_data, hps=pca_hps), L2Normalize()]
# dim_process = [PCA(train_fea=train_data, hps=pca_hps)]
dim_process = [L2Normalize()]

################ feature_enhancer for gallery
feature_enhancer = [DBA({"enhance_k": 10}, batch_size=2000)]

############## train done!

################## set metric_agent

metric_agent = MetricAgent()
# metric_agent.set_dim_process(dim_process)
# metric_agent.set_fea_enhancer(feature_enhancer)
metric_agent.set_gallery(gallery=gallery, gallery_label=gallery_label)

# ################## infer query and rewrite json ################
# print("start infering query_data.")
# cnt = 0
# with torch.no_grad():
#     for roi_tensor_batch, label_tensor_batch in tqdm(query_generator):
#         feat = model(roi_tensor_batch.to(device))
#         feat = aggregator(feat)
#         for pred in metric_agent.run_metric(feat.detach().cpu().numpy()):
#             # rewrite
#             annotation_id = int(query_data.samples[cnt][0].split("_id")[-1].split('.png')[0])
#             json_file["annotations"][annotation_id]["category_id"] = int(pred)
#             cnt += 1
# metric_agent.reset_gallery()
# with open("./submit_rewrite.json", "w") as f:
#     json.dump(json_file, f)


################## infer query and rewrite json ################
############## infer query in one run_metric and use dba
print("start infering query_data.")
rewrite_test_query_features = False
if rewrite_test_query_features:
    test_query = []
    with torch.no_grad():
        for roi_tensor_batch, label_tensor_batch in tqdm(query_generator):
            feat = model(roi_tensor_batch.to(device))
            feat = aggregator(feat)
            # _, feat, _ = model(roi_tensor_batch.to(device))
            # feat, _, _ = model(roi_tensor_batch.to(device))
            # feat = F.softmax(feat, dim=-1)
            test_query.append(feat.detach())
    test_query = torch.cat(test_query, dim=0)
    test_query = test_query.cpu().numpy()
    np.save("test_query.npy", test_query)
else:
    test_query = np.load("test_query.npy")

cnt = 0
match_cnt = 0

dismatch_list = []  # (id, image_path, best_pred, pred)
for pred in metric_agent.run_metric(test_query):
    # rewrite
    annotation_id = int(query_data.samples[cnt][0].split("_id")[-1].split('.png')[0])
    if not json_file["annotations"][annotation_id]["category_id"] == int(pred):
        image_name, _ = query_data.samples[cnt]
        dismatch_list.append(
            (cnt, query_data.root + image_name, json_file["annotations"][annotation_id]["category_id"], int(pred)))
    if json_file["annotations"][annotation_id]["category_id"] == int(pred):
        match_cnt += 1
    json_file["annotations"][annotation_id]["category_id"] = int(pred)
    cnt += 1

metric_agent.reset_gallery()
with open("./submit_rewrite.json", "w") as f:
    json.dump(json_file, f)
print("match rate: {}%".format(match_cnt * 100.0 / cnt))
print("rewrite json file done!")

# save pic
# dismatch_save = "./dismatch"
# if os.path.exists(dismatch_save):
#     os.remove(dismatch_save)
# os.makedirs(dismatch_save)

shuffle(dismatch_list)
for dismatch in dismatch_list:
    idx, image_path, best_pred, pred = dismatch
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    cv2.imshow("1", image)
    print(idx, best_pred, pred)
    cv2.waitKey()
