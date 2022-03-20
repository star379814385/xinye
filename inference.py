# coding=utf-8

import os
import sys

sys.path[0] = os.path.abspath(os.path.dirname(__file__)) + "/tools"
from mmdet.apis import inference_detector, init_detector, show_result_pyplot

sys.path[0] = os.path.abspath(os.path.dirname(__file__))

import cv2
from tqdm import tqdm
import json

from tools.retrieval_train import DataLoader, QuiryDataset, KeyDataset, Model
from utils.data.image_crop import crop_img_test, crop_img
import glob
import torch
from tools.pyretri.extract.aggregator import GAP
from tools.pyretri.index.dim_processor import PCA, L2Normalize
from tools.pyretri.index.feature_enhancer import DBA
from tools.retrieval_runner.metric_agent import MetricAgent
from tools.pyretri.index.re_ranker import QEKR, QE
import numpy as np
from utils.func import num_clip
from albumentations import Compose, Resize, Normalize, DualTransform

# from albumentations import *
from copy import deepcopy


#
class PadResize(DualTransform):

    def __init__(self, size, const_value=(0, 0, 0), interpolation=cv2.INTER_LINEAR, always_apply=False, p=1):
        super(PadResize, self).__init__(always_apply, p)
        self.size = size
        self.interpolation = interpolation
        self.background = np.array([[const_value]], dtype=np.uint8).repeat(self.size, 0).repeat(self.size, 1)

    def apply(self, img, interpolation=cv2.INTER_LINEAR, **params):
        target_size = self.size
        interpolation = self.interpolation
        w, h = img.shape[:2]
        img_xx = cv2.resize(img, (self.size, self.size))
        if w > h:
            img = cv2.resize(img, (target_size, int(h * target_size * 1.0 / w)), interpolation=interpolation)
        else:
            img = cv2.resize(img, (int(w * target_size * 1.0 / h), int(target_size)), interpolation=interpolation)

        ret_img = deepcopy(self.background)
        h, w = img.shape[:2]
        st_h = int((ret_img.shape[0] - h) / 2.0)
        st_w = int((ret_img.shape[1] - w) / 2.0)
        ret_img[st_h: st_h + h, st_w: st_w + w] = img

        # ret_img = np.hstack([ret_img, img_xx])
        return ret_img

        # return F.resize(img, height=self.height, width=self.width, interpolation=interpolation)

    def apply_to_bbox(self, bbox, **params):
        # Bounding box coordinates are scale invariant
        return bbox

    def apply_to_keypoint(self, keypoint, **params):
        height = params["rows"]
        width = params["cols"]
        scale_x = self.width / width
        scale_y = self.height / height
        return F.keypoint_scale(keypoint, scale_x, scale_y)

    def get_transform_init_args_names(self):
        return ("height", "width", "interpolation")


# def run():
#     annotations_info = {'images': [], 'annotations': []}
#     SAVE_JSON_PATH = "./submit/output.json"
#     print(len(annotations_info["annotations"]))
#     with open(SAVE_JSON_PATH, "w") as f:
#         json.dump(annotations_info, f)


def run():
    # load_model_weights()
    # preprocessing()
    # inference()
    # save_result()

    # base_path = os.path.abspath(os.path.dirname(__file__))
    # os.chdir(base_path)

    # print(base_path)

    """
    ##############   1 根据testB标注裁剪roi保存本地
    """
    # image_crop for testB

    crop_img(json_path="./project_src/Project_test/data/test/b_annotations.json",
             data_root="./project_src/Project_test/data/test/b_images",
             save_dir="./project_src/Project_test/data/test/b_images_crop")
    # crop_img_test(json_path="./project_src/Project_test/data/test/b_annotations.json",
    #               data_root="./project_src/Project_test/data/test/b_images",
    #               save_dir="./project_src/Project_test/data/test/b_images_crop")

    """
    #############   2 相关推断配置
    #############   2.1 目标检测配置
    """
    # 模型配置文件
    # config_file = "./model/detection/configs/cascade_mask_rcnn_swin_base_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py"
    config_file = "./project_src/Project_test/model/detection/configs/cascade_mask_rcnn_swin_base_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py"
    # 预训练模型文件
    # checkpoint_file = './model_files/detection/cascade_mask_rcnn_swin_base_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco/epoch_20.pth'
    checkpoint_file = './project_src/Project_test/model_files/detection/cascade_mask_rcnn_swin_base_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco/epoch_20.pth'
    # 通过模型配置文件与预训练文件构建模型
    model_detect = init_detector(config_file, checkpoint_file, device='cuda:0')

    # TRAIN_DATA_ROOT = "./data/初赛数据集/train/"
    # TEST_DATA_ROOT = "./data/test/"
    TEST_DATA_ROOT = "./project_src/Project_test/data/test/"

    # test_json_file = json.load(open(TRAIN_DATA_ROOT + "a_annotations.json", "r"))
    test_json_file = json.load(open(TEST_DATA_ROOT + "a_annotations.json", "r"))
    """
    ############    2.2 检索配置
    """
    # retrieval config
    roi_size = (224, 224)
    ############# use_augs in gallery
    gallery_aug = True
    # gallery_data = KeyDataset(root="./data/test/b_images_crop", width=roi_size[0], height=roi_size[1],
    #                           use_augs=gallery_aug)
    gallery_data = KeyDataset(root="./project_src/Project_test/data/test/b_images_crop", width=roi_size[0],
                              height=roi_size[1],
                              use_augs=gallery_aug)
    gallery_generator = DataLoader(dataset=gallery_data, batch_size=16, shuffle=False, num_workers=0,
                                   collate_fn=gallery_data.collate_fn)

    #################### load model and load weight
    # model_name = "se_resnet50"
    # model_name = "resnext50_32x4d"
    model_retrieval_name = "efficientnet-b5"
    # model_retrieval_name = "swin_b224"
    # model_retrieval_name = "volo_d3"
    # model_retri = Model(model_name=model_retrieval_name, num_classes=116, neck="bnneck").cuda()
    after_aggre = True
    model_retri = Model(model_name=model_retrieval_name, num_classes=116, neck="no").cuda()
    if not after_aggre:
        model_retri = model_retri.base
    # model_cls = None
    weight_path = None
    if weight_path is None:
        # weight_path = glob.glob("./model_files/retrieval/{}/model/best*.pth".format(model_retrieval_name))
        weight_path = glob.glob(
            "./project_src/Project_test/model_files/retrieval/{}/model/best*.pth".format(model_retrieval_name))
        assert len(weight_path) == 1
        weight_path = weight_path[0]
    model_retri.load_state_dict(torch.load(weight_path))
    print("model load weight from " + weight_path)
    ########### just use backbone for extring features
    # model_retri = model_retri.base
    model_retri = model_retri.eval()
    ##########  set parallel
    Parallel = False
    model_retri = torch.nn.DataParallel(model_retri,
                                        device_ids=range(0, torch.cuda.device_count(), 1)) if Parallel else model_retri
    device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    ############### change gap to other aggregator
    agg_list = [GAP()]
    assert len(agg_list) > 0
    aggregator = lambda x: torch.cat([agg({"fea": x})["fea_" + agg.__class__.__name__] for agg in agg_list], dim=-1)
    print("using aggregator as follow:")
    print([a.__class__.__name__ for a in agg_list])

    """
    ###########     3 推断testB的roi获取检索库特征
    """
    ############### load gallery features ###########
    print("getting gallery and gallery_label from infering key_data.")
    rewrite_test_gallery_features = True
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
                if after_aggre:
                    _, feat, _ = model_retri(img_tensor)
                else:
                    feat = model_retri(img_tensor)
                    feat = aggregator(feat)
                # to numpy
                gallery_list.append(feat.detach().cpu().numpy())
                gallery_label_list.append(label_tensor.detach().cpu().numpy())
        gallery = np.concatenate(gallery_list, axis=0)
        gallery_label = np.concatenate(gallery_label_list, axis=0)
    #     np.save("test_gallery.npy", gallery)
    #     np.save("test_gallery_label.npy", gallery_label)
    # else:
    #     gallery = np.load("test_gallery.npy")
    #     gallery_label = np.load("test_gallery_label.npy")
    """
        ############    4 设定检索配置，初始化检索器
    """
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
    feature_enhancer = [DBA({"enhance_k": 1}, batch_size=2000)]

    ############## train done!

    ################## set metric_agent

    retri_agent = MetricAgent()
    # metric_agent.set_dim_process(dim_process)
    # retri_agent.set_fea_enhancer(feature_enhancer)
    retri_agent.set_gallery(gallery=gallery, gallery_label=gallery_label)

    query_features = []
    query_pred = []

    """
    ############    5 目标检测获取目标，并使用检索方法更改预测结果
    """
    transforms = Compose([
        # Resize(roi_size[0], roi_size[1], p=1),
        PadResize(size=roi_size[0], p=1),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1),
    ])
    annotations_info = {'images': [], 'annotations': []}
    roi_batch = []
    anno_batch = []
    batch_size = 64
    # batch_size = 8
    features_id = 0
    test_json_file["images"].append([1])
    for image_cnt, image in enumerate(tqdm(test_json_file["images"])):
        # 不知道为何最后一图出bug
        if image_cnt == len(test_json_file["images"]) - 1:
            break
        # img = cv2.imread(TRAIN_DATA_ROOT + "/a_images/" + image["file_name"], -1)
        img = cv2.imread(TEST_DATA_ROOT + "/a_images/" + image["file_name"], -1)
        img_h, img_w = img.shape[:2]
        # predictions, vis_output = demo.run_on_image(img)
        predictions = inference_detector(model_detect, img)

        # 可视化
        # show_result_pyplot(model, img, predictions, score_thr=0.05)

        # 生成submit
        annotations_info["images"].append({"file_name": image["file_name"], "id": image["id"]})
        for cls_index, cls_predictions in enumerate(predictions):
            for prediction in cls_predictions:
                if prediction[-1] < 0.05:
                    continue
                x1, y1, x2, y2 = prediction[:4]
                x = float(x1)
                y = float(y1)
                w = float(x2 - x1)
                h = float(y2 - y1)

                # 获取roi特征
                x1 = num_clip(x1, 0, img_w)
                x2 = num_clip(x2, 0, img_w)
                y1 = num_clip(y1, 0, img_h)
                y2 = num_clip(y2, 0, img_h)
                roi = img[int(y1):int(y2), int(x1):int(x2), :]
                roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                roi = transforms(image=roi)["image"]

                roi_batch.append(roi)
                anno_batch.append({"image_id": image["id"],
                                   "bbox": [x, y, w, h],
                                   "category_id": cls_index,
                                   "score": float(prediction[4])}
                                  )
                if len(roi_batch) >= batch_size:
                    roi_tensor = torch.tensor(np.stack(roi_batch).transpose(0, 3, 1, 2), dtype=torch.float32)
                    with torch.no_grad():
                        # roi_fea = model_retri(roi_tensor.to(device))
                        # roi_fea = aggregator(roi_fea)

                        if after_aggre:
                            _, roi_fea, _ = model_retri(roi_tensor.to(device))
                        else:
                            roi_fea = model_retri(roi_tensor.to(device))
                            roi_fea = aggregator(roi_fea)
                    roi_fea = roi_fea.detach().cpu().numpy()
                    # 更改类别预测
                    for i, cls_index in enumerate(retri_agent.run_metric(roi_fea)):
                        anno_batch[i]["category_id"] = int(cls_index)
                        # 保存query_features,query_pred
                        anno_batch[i]["features_id"] = features_id
                        features_id += 1
                        query_features.append(roi_fea[i])
                        query_pred.append(int(cls_index))
                    annotations_info["annotations"] += anno_batch
                    roi_batch = []
                    anno_batch = []

        if image_cnt + 1 >= len(test_json_file["images"]) and len(roi_batch) > 0:
            roi_tensor = torch.tensor(np.stack(roi_batch).transpose(0, 3, 1, 2), dtype=torch.float32)
            with torch.no_grad():
                if after_aggre:
                    _, roi_fea, _ = model_retri(roi_tensor.to(device))
                else:
                    roi_fea = model_retri(roi_tensor.to(device))
                    roi_fea = aggregator(roi_fea)
            roi_fea = roi_fea.detach().cpu().numpy()
            # 更改类别预测
            for i, cls_index in enumerate(retri_agent.run_metric(roi_fea)):
                anno_batch[i]["category_id"] = int(cls_index)
                # 保存query_features,query_pred
                anno_batch[i]["features"] = roi_fea[i]
                anno_batch[i]["features_id"] = features_id
                features_id += 1
                query_features.append(roi_fea[i])
                query_pred.append(int(cls_index))
            annotations_info["annotations"] += anno_batch
            roi_batch = []
            anno_batch = []

    # 获取query特征
    query_features = np.stack(query_features, axis=0)
    query_pred = np.array(query_pred)

    # 1.将原有query特征加入gellery, gallery_enhence
    # retri_agent.add_gallery(query_features, query_pred)
    # retri_agent.k = 5
    # query_pred_gellery_enhence = retri_agent.run_metric(query_features)
    # query_pred = query_pred

    # 1.将原有query特征加入gellery, gallery_enhence
    # 增加循环更新gallery
    num_gallery_enhence = 3
    retri_agent.k = 6
    for i in range(num_gallery_enhence):
        retri_agent.set_gallery(
            np.concatenate([gallery, query_features], axis=0),
            np.concatenate([gallery_label, query_pred], axis=0)
        )

        query_pred = retri_agent.run_metric(query_features)
    query_pred_gellery_enhence = query_pred

    # 2.尝试qe
    gallery = np.concatenate([gallery, query_features], axis=0)
    gallery_label = np.concatenate([gallery_label, query_pred], axis=0)
    qe_hps = {"qe_times": 1, "qe_k": 6}
    qe_runner = QE(hps=qe_hps)

    # query_features = torch.tensor(query_features, dtype=torch.float32)
    gallery_features = torch.tensor(gallery, dtype=torch.float32).to(device)
    # sorted_idx = qe_runner._cal_dis(query_fea=query_features, gallery_fea=gallery_features)

    # gallery_features = gallery_features.cuda()

    sorted_idx_list = []
    n = 0
    unit_size = 200
    while(n < query_features.shape[0]):
        query_features_unit = query_features[n: n+unit_size]
        query_features_unit = torch.tensor(query_features_unit, dtype=torch.float32).to(device)
        sorted_idx_unit = qe_runner._cal_dis(query_fea=query_features_unit, gallery_fea=gallery_features)
        sorted_idx_unit = sorted_idx_unit.cpu().long()
        sorted_idx_unit = qe_runner(query_fea=query_features_unit, gallery_fea=gallery_features, sorted_index=sorted_idx_unit)
        sorted_idx_unit = sorted_idx_unit.cpu().numpy()
        sorted_idx_list.append(sorted_idx_unit)
        n += unit_size
    sorted_idx = np.concatenate(sorted_idx_list, axis=0)


    # sorted_idx = qe_runner(query_fea=query_features, gallery_fea=gallery_features, sorted_index=sorted_idx)
    # sorted_idx = sorted_idx.detach().cpu().numpy()
    # just get max
    # sorted_idx = sorted_idx[:, :1]
    pred_cls_ids = np.array(list(map(lambda x: gallery_label[x], sorted_idx)), dtype=np.int32)
    query_pred_query_expansion = np.array(list(map(lambda x: np.argmax(np.bincount(x)), pred_cls_ids)), dtype=np.int32)
    #
    # query_pred_post = query_pred_gellery_enhence
    query_pred_post = query_pred_query_expansion

    # 再次更新标注预测
    for idx in range(len(annotations_info["annotations"])):
        fea_id = annotations_info["annotations"][idx]["features_id"]
        annotations_info["annotations"][idx]["category_id"] = int(query_pred_post[fea_id])
        del annotations_info["annotations"][idx]["features_id"]

    # SAVE_JSON_PATH = "./submit/output.json"
    SAVE_JSON_PATH = "./project_src/Project_test/submit/output.json"
    # print(len(annotations_info["annotations"]))
    with open(SAVE_JSON_PATH, "w") as f:
        json.dump(annotations_info, f)

    return


if __name__ == "__main__":
    import time

    a = time.time()
    run()
    b = time.time()
    print((b - a) / 60)
