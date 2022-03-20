import json
import numpy as np

num_classes = 116


def cul_hist(annotations):
    hist = np.array([0] * num_classes, dtype=np.int32)
    for a in annotations:
        hist[a["category_id"]] += 1
    return hist


def cul_overlap(arrayA, arrayB):
    assert arrayA.shape == arrayB.shape and arrayA.dtype == arrayB.dtype
    overlap = np.bitwise_and(arrayA.astype(np.bool), arrayB.astype(np.bool))
    return np.sum(overlap * arrayA) * 1.0 / np.sum(arrayA)


if __name__ == "__main__":
    trainA = json.load(open("./初赛数据集/train/a_annotations.json", "r"))
    trainB = json.load(open("./初赛数据集/train/b_annotations.json", "r"))
    # testA = json.load(open("./初赛数据集/train/a_annotations.json", "r"))
    # testB = json.load(open("./初赛数据集/train/a_annotations.json", "r"))

    trainA_annotations_hist = cul_hist(trainA["annotations"])
    trainB_annotations_hist = cul_hist(trainB["annotations"])
    print(trainA_annotations_hist)
    print(trainB_annotations_hist)
    # overlap = cul_overlap(trainA_annotations_hist, trainB_annotations_hist)
    # print(trainA_annotations_hist)
    # print(trainB_annotations_hist)
    # print(overlap)  # 0.9589702333065165
