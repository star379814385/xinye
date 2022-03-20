import sys
import cv2
import json
import os
from tqdm import tqdm


def crop_img(json_path, data_root, save_dir):
    json_file = json.load(open(json_path, "r"))
    for anno in tqdm(json_file["annotations"]):
        assert anno["image_id"] == json_file["images"][anno["image_id"]]["id"]

        img_name = json_file["images"][anno["image_id"]]["file_name"]
        img = cv2.imread(data_root + "/" + img_name, -1)
        assert len(img.shape) == 3
        bbox = anno["bbox"]
        img_crop = img[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2], :]
        cls_id = anno["category_id"]
        assert cls_id == json_file["categories"][cls_id]["id"]
        save_file = "{}/{}_{}".format(save_dir, cls_id, json_file["categories"][cls_id]["name"])
        save_name = img_name.replace(".jpg", "_id{}.png".format(anno["id"]))

        if not os.path.exists(save_file):
            os.makedirs(save_file)

        cv2.imwrite(save_file + "/" + save_name, img_crop)


def crop_img_test(json_path, data_root, save_dir):

    json_file = json.load(open(json_path, "r"))
    json_file["categories"] = json.load(open("/media/gdut502/139e9283-5fa3-498b-ba3d-0272a299eeeb/wqr/xinye/xinye2021"
                                             "/data/test/b_annotations.json", "r"))["categories"]
    print(json_file["categories"])
    id = 0
    for anno in tqdm(json_file["annotations"]):
        assert anno["image_id"] == json_file["images"][anno["image_id"]]["id"]

        img_name = json_file["images"][anno["image_id"]]["file_name"]
        img = cv2.imread(data_root + "/" + img_name, -1)
        assert len(img.shape) == 3
        bbox = [int(bbox) for bbox in anno["bbox"]]
        img_crop = img[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2], :]
        cls_id = anno["category_id"]
        assert cls_id == json_file["categories"][cls_id]["id"]
        save_file = "{}/{}_{}".format(save_dir, cls_id, json_file["categories"][cls_id]["name"])
        save_name = img_name.replace(".jpg", "_id{}.png".format(id))
        id += 1

        if not os.path.exists(save_file):
            os.makedirs(save_file)

        cv2.imwrite(save_file + "/" + save_name, img_crop)


# scale
def crop_img2(json_path, data_root, save_dir, scale=1.2):
    json_file = json.load(open(json_path, "r"))
    for anno in tqdm(json_file["annotations"]):
        assert anno["image_id"] == json_file["images"][anno["image_id"]]["id"]

        img_name = json_file["images"][anno["image_id"]]["file_name"]
        img = cv2.imread(data_root + "/" + img_name, -1)
        img_h, img_w = img.shape[:2]
        assert len(img.shape) == 3
        bbox = anno["bbox"]
        x, y, w, h = bbox
        x_c, y_c = x + w / 2, y + h / 2
        w, h = int(w * scale), int(h * scale)
        x1, y1 = x_c - w / 2, y_c - h / 2
        x2, y2 = x_c + w / 2, y_c + h / 2
        x1 = int(x1) if x1 > 0 else 0
        y1 = int(y1) if y1 > 0 else 0
        x2 = int(x2) if x2 <= img_w else img_w
        y2 = int(y2) if y2 <= img_h else img_h
        img_crop = img[y1:y2, x1:x2, :]
        cls_id = anno["category_id"]
        assert cls_id == json_file["categories"][cls_id]["id"]
        # save_file = "{}/{}_{}".format(save_dir, cls_id, json_file["categories"][cls_id]["name"])
        save_file = "{}_s{}/{}_{}".format(save_dir, scale, cls_id, json_file["categories"][cls_id]["name"])
        save_name = img_name.replace(".jpg", "_id{}.png".format(anno["id"]))

        if not os.path.exists(save_file):
            os.makedirs(save_file)

        cv2.imwrite(save_file + "/" + save_name, img_crop)


if __name__ == "__main__":

    phase, data_type = sys.argv[1], sys.argv[2]
    assert phase in ["train", "test"]
    assert data_type in ["a", "b"]
    # phase = "train"
    # data_type = "a"
    # exit()

    # train
    # data_root = "../../data/{}/{}_images".format(phase, data_type)
    # json_path = "../../{}/{}_annotations.json".format(phase, data_type)
    data_root = "./data/{}/{}_images".format(phase, data_type)
    json_path = "./data/{}/{}_annotations.json".format(phase, data_type)
    save_dir = "./data/{}/{}_images_crop".format(phase, data_type)
    if phase == "test" and data_type == "a":
        json_path = "./submit/detection_submit.json"
        # crop_img_test(json_path, data_root, save_dir=save_dir)
        crop_img_test(json_path, data_root, save_dir=save_dir)
    else:
        crop_img(json_path, data_root, save_dir=save_dir)
        # crop_img2(json_path, data_root, save_dir=save_dir, scale=1.2)
