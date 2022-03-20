import json
from random import shuffle
from tqdm import tqdm


def split_json(save_root="./", train_p=0.8, is_shuffle=True):
    # data A
    # json_file = json.load(open("./初赛数据集/train/a_annotations.json", "r"))
    json_file = json.load(open("./train/a_annotations.json", "r"))
    # print([cls["name"] for cls in json_file["categories"]])
    # exit()
    images = json_file["images"]
    dataA_len = len(images)
    annotationsA_len = len(json_file["annotations"])


    shuffle(images) if is_shuffle else None

    train_annotations_info = {'images': [], 'annotations': [], 'categories': []}
    val_annotations_info = {'images': [], 'annotations': [], 'categories': []}

    train_num = int(len(images) * train_p)
    train_annotations_info["images"], val_annotations_info["images"] = images[:train_num], images[train_num:]
    # train
    for image in tqdm(train_annotations_info["images"]):
        rest_annotations = []
        for anno in json_file["annotations"]:
            if anno["image_id"] == image["id"]:
                train_annotations_info["annotations"].append(anno)
            else:
                rest_annotations.append(anno)
        json_file["annotations"] = rest_annotations
    # val
    for image in tqdm(val_annotations_info["images"]):
        rest_annotations = []
        for anno in json_file["annotations"]:
            if anno["image_id"] == image["id"]:
                val_annotations_info["annotations"].append(anno)
            else:
                rest_annotations.append(anno)
        json_file["annotations"] = rest_annotations
    assert len(json_file["annotations"]) == 0

    # data B
    # json_file = json.load(open("./初赛数据集/train/b_annotations.json", "r"))
    json_file = json.load(open("./train/b_annotations.json", "r"))
    images = json_file["images"]
    annotations = json_file["annotations"]
    for i in range(len(images)):
        images[i]["id"] += dataA_len
        images[i]["file_name"] = "../b_images/" + images[i]["file_name"]
    for i in range(len(annotations)):
        annotations[i]["image_id"] += dataA_len
        annotations[i]["id"] += annotationsA_len
    train_annotations_info["images"] += images
    train_annotations_info["annotations"] += annotations


    train_annotations_info["categories"] = json_file["categories"]
    val_annotations_info["categories"] = json_file["categories"]

    # logger
    print("Train dataset: exist {} images，{} annotations".format(
        len(train_annotations_info["images"]), len(train_annotations_info["annotations"])))
    print("Val dataset: exist {} images，{} annotations".format(
        len(val_annotations_info["images"]), len(val_annotations_info["annotations"])))

    # write
    with open(save_root + "train_ab_all.json", "w") as f:
        json.dump(train_annotations_info, f)

    with open(save_root + "val_ab_all.json", "w") as f:
        json.dump(val_annotations_info, f)




    # shuffle(images)
    # print(images)


if __name__ == "__main__":
    # split_json(save_root="./初赛数据集/train/", train_p=1.0)
    import os
    os.chdir(os.path.dirname(__file__) + "/../../data/")

    split_json(save_root="./train/", train_p=1.0)
    split_json(save_root="./test/", train_p=1.0)
