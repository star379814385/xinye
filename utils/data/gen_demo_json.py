import json
from random import shuffle
from tqdm import tqdm


def split_json(json_path, save_root="./", p=0.01, is_shuffle=True):
    json_file = json.load(open(json_path, "r"))
    # print([cls["name"] for cls in json_file["categories"]])
    # exit()
    images = json_file["images"]

    shuffle(images) if is_shuffle else None

    train_annotations_info = {'images': [], 'annotations': [], 'categories': []}
    val_annotations_info = {'images': [], 'annotations': [], 'categories': []}

    train_num = int(len(images) * p)
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

    train_annotations_info["categories"] = json_file["categories"]
    # logger
    print("Train dataset: exist {} images，{} annotations".format(
        len(train_annotations_info["images"]), len(train_annotations_info["annotations"])))

    # write
    with open(save_root + "demo.json", "w") as f:
        json.dump(train_annotations_info, f)




    # shuffle(images)
    # print(images)


if __name__ == "__main__":
    split_json(json_path="./初赛数据集/train/a_annotations.json", save_root="./初赛数据集/train/")
