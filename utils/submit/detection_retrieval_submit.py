# coding=utf-8
import sys
import os

sys.path[0] = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))+"/tools"

from mmdet.apis import inference_detector, init_detector, show_result_pyplot
import cv2
from tqdm import tqdm
import json

from tools.retrieval_train import DataLoader, QuiryDataset, KeyDataset, Model
from utils.data.image_crop import crop_img_test
import glob
import torch
from tools.pyretri.extract.aggregator import GAP
from tools.pyretri.index.dim_processor import PCA, L2Normalize
from tools.pyretri.index.feature_enhancer import DBA
from tools.retrieval_runner.metric_agent import MetricAgent
import numpy as np
from utils.func import num_clip
from albumentations import Compose, Resize, Normalize


if __name__ == "__main__":
    """
    ##############   1 根据testB标注裁剪roi保存本地
    """
    # image_crop for testB
    crop_img_test(json_path="../../data/test/b_annotations.json",
                  data_root="../../data/test/b_images",
                  save_dir="../../data/test/b_images_crop")

    """
    #############   2 相关推断配置
    #############   2.1 目标检测配置
    """
    # # 模型配置文件
    # config_file = "../../model/detection/configs/cascade_mask_rcnn_swin_base_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py"
    # # 预训练模型文件
    # checkpoint_file = '../../model_files/detection/cascade_mask_rcnn_swin_base_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco/epoch_20.pth'
    # 模型配置文件
    config_file = "../../model_files/detection/single_detection/single_detection.py"
    # 预训练模型文件
    checkpoint_file = '../../model_files/detection/single_detection/epoch_18.pth'
    # 通过模型配置文件与预训练文件构建模型
    model_detect = init_detector(config_file, checkpoint_file, device='cuda:0')

    # TRAIN_DATA_ROOT = "./data/初赛数据集/train/"
    TEST_DATA_ROOT = "../../data/test/"

    # test_json_file = json.load(open(TRAIN_DATA_ROOT + "a_annotations.json", "r"))
    test_json_file = json.load(open(TEST_DATA_ROOT + "a_annotations.json", "r"))
    """
    ############    2.2 检索配置
    """
    # retrieval config
    roi_size = (224, 224)
    ############# use_augs in gallery
    gallery_aug = True
    gallery_data = KeyDataset(root="../../data/test/b_images_crop", width=roi_size[0], height=roi_size[1],
                              use_augs=gallery_aug)
    gallery_generator = DataLoader(dataset=gallery_data, batch_size=64, shuffle=False, num_workers=16,
                                   collate_fn=gallery_data.collate_fn)
    #################### load model and load weight
    # model_name = "se_resnet50"
    # model_name = "resnext50_32x4d"
    # model_name = "efficientnet-b5"
    model_retrieval_name = "swin_b224"
    # model = Model(model_name=model_name, num_classes=116, neck="bnneck").cuda()
    model_retri = Model(model_name=model_retrieval_name, num_classes=116, neck="no").cuda()
    # model_cls = None
    weight_path = None
    if weight_path is None:
        weight_path = glob.glob("/media/gdut502/139e9283-5fa3-498b-ba3d-0272a299eeeb/wqr/xinye/xinye2021/model_files"
                                "/retrieval/{}/model/best*.pth".format(model_retrieval_name))
        assert len(weight_path) == 1
        weight_path = weight_path[0]
    model_retri.load_state_dict(torch.load(weight_path))
    print("model load weight from " + weight_path)
    ########### just use backbone for extring features
    model_retri = model_retri.base
    model_retri = model_retri.eval()
    ##########  set parallel
    Parallel = False
    model_retri = torch.nn.DataParallel(model_retri, device_ids=range(0, torch.cuda.device_count(), 1)) if Parallel else model_retri
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
                feat = model_retri(img_tensor)
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
    feature_enhancer = [DBA({"enhance_k": 10}, batch_size=2000)]

    ############## train done!

    ################## set metric_agent

    retri_agent = MetricAgent()
    # metric_agent.set_dim_process(dim_process)
    # metric_agent.set_fea_enhancer(feature_enhancer)
    retri_agent.set_gallery(gallery=gallery, gallery_label=gallery_label)

    """
    ############    5 目标检测获取目标，并使用检索方法更改预测结果
    """
    transforms = Compose([
        Resize(roi_size[0], roi_size[1], p=1),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1),
    ])
    annotations_info = {'images': [], 'annotations': []}
    for image in tqdm(test_json_file["images"]):
        # img = cv2.imread(TRAIN_DATA_ROOT + "/a_images/" + image["file_name"], -1)
        img = cv2.imread(TEST_DATA_ROOT + "/a_images/" + image["file_name"], -1)
        # cv2.imshow("a", img)
        # cv2.waitKey()
        # exit()
        img_h, img_w = img.shape[:2]
        # predictions, vis_output = demo.run_on_image(img)
        predictions = inference_detector(model_detect, img)
        # 可视化
        # show_result_pyplot(model_detect, img, predictions, score_thr=0.05)

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
                roi_tensor = torch.tensor(roi[None].transpose(0, 3, 1, 2), dtype=torch.float32)
                with torch.no_grad():
                    roi_fea = model_retri(roi_tensor.to(device))
                    print(roi_fea.shape)
                    roi_fea = aggregator(roi_fea)
                    print(roi_fea.shape)
                roi_fea = roi_fea.detach().cpu().numpy()
                # 更改类别预测
                cls_index = retri_agent.run_metric(roi_fea)[0]
                annotations_info["annotations"].append({"image_id": image["id"],
                                                        "bbox": [x, y, w, h],
                                                        "category_id": cls_index,
                                                        "score": float(prediction[4])
                                                        })
        break

    SAVE_JSON_PATH = "../../submit/detection_submit.json"
    with open(SAVE_JSON_PATH, "w") as f:
        json.dump(annotations_info, f)