# coding=utf-8
import sys
import os

sys.path[0] = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))+"/tools"

from mmdet.apis import inference_detector, init_detector, show_result_pyplot
import cv2
from tqdm import tqdm
import json



if __name__ == "__main__":
    # object config
    # 模型配置文件
    config_file = "../../model/detection/configs/single_detection.py"
    # 预训练模型文件
    checkpoint_file = '../../model_files/detection/single_detection/epoch_18.pth'
    # 通过模型配置文件与预训练文件构建模型
    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    TEST_DATA_ROOT = "../../data/test/"
    test_json_file = json.load(open(TEST_DATA_ROOT + "a_annotations.json", "r"))


    annotations_info = {'images': [], 'annotations': []}
    for image in tqdm(test_json_file["images"]):
        # img = cv2.imread(TRAIN_DATA_ROOT + "/a_images/" + image["file_name"], -1)
        img = cv2.imread(TEST_DATA_ROOT + "/a_images/" + image["file_name"], -1)
        # predictions, vis_output = demo.run_on_image(img)
        predictions = inference_detector(model, img)

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

                annotations_info["annotations"].append({"image_id": image["id"],
                                                        "bbox": [x, y, w, h],
                                                        "category_id": cls_index,
                                                        "score": float(prediction[4])
                                                        })

    SAVE_JSON_PATH = "../../submit/detection_submit.json"
    with open(SAVE_JSON_PATH, "w") as f:
        json.dump(annotations_info, f)