import os
import cv2
import json
import xml.dom.minidom
from tqdm import tqdm
import xml.etree.ElementTree as ET
from random import shuffle
from copy import deepcopy


def xml2json(data_dir='./data/train', write_dir='./annotations.json'):
    # data_dir = './data/train' #根目录文件，其中包含image文件夹和box文件夹（根据自己的情况修改这个路径）

    image_file_dir = os.path.join(data_dir, 'image')
    xml_file_dir = os.path.join(data_dir, 'box')

    annotations_info = {'images': [], 'annotations': [], 'categories': []}

    categories_map = {"__background__":0, 'holothurian': 1, 'echinus': 2, 'scallop': 3, 'starfish': 4}

    for key in categories_map:
        categoriy_info = {"id":categories_map[key], "name":key}
        annotations_info['categories'].append(categoriy_info)

    file_names = [image_file_name.split('.')[0]
                  for image_file_name in os.listdir(image_file_dir)]
    ann_id = 1
    for i, file_name in enumerate(tqdm(file_names)):
        # 去除困难样本
        if "u" in file_name:
            continue
        image_file_name = file_name + '.jpg'
        xml_file_name = file_name + '.xml'
        image_file_path = os.path.join(image_file_dir, image_file_name)
        xml_file_path = os.path.join(xml_file_dir, xml_file_name)

        image_info = dict()
        image = cv2.cvtColor(cv2.imread(image_file_path), cv2.COLOR_BGR2RGB)
        height, width, _ = image.shape
        image_info = {'file_name': image_file_name, 'id': i+1,
                      'height': height, 'width': width}
        annotations_info['images'].append(image_info)

        DOMTree = xml.dom.minidom.parse(xml_file_path)
        collection = DOMTree.documentElement

        names = collection.getElementsByTagName('name')
        names = [name.firstChild.data for name in names]

        xmins = collection.getElementsByTagName('xmin')
        xmins = [xmin.firstChild.data for xmin in xmins]
        ymins = collection.getElementsByTagName('ymin')
        ymins = [ymin.firstChild.data for ymin in ymins]
        xmaxs = collection.getElementsByTagName('xmax')
        xmaxs = [xmax.firstChild.data for xmax in xmaxs]
        ymaxs = collection.getElementsByTagName('ymax')
        ymaxs = [ymax.firstChild.data for ymax in ymaxs]

        object_num = len(names)

        for j in range(object_num):
            if names[j] in categories_map:
                image_id = i + 1
                x1,y1,x2,y2 = int(xmins[j]),int(ymins[j]),int(xmaxs[j]),int(ymaxs[j])
                x1,y1,x2,y2 = x1 - 1,y1 - 1,x2 - 1,y2 - 1

                if x2 == width:
                    x2 -= 1
                if y2 == height:
                    y2 -= 1

                x,y = x1,y1
                w,h = x2 - x1 + 1,y2 - y1 + 1
                category_id = categories_map[names[j]]
                area = w * h
                annotation_info = {"id": ann_id, "image_id":image_id, "bbox":[x, y, w, h], "category_id": category_id, "area": area,"iscrowd": 0}
                annotations_info['annotations'].append(annotation_info)
                ann_id += 1

    with open(write_dir, 'w') as f:
        json.dump(annotations_info, f, indent=4)

    print('---整理后的标注文件---')
    print('所有图片的数量：',  len(annotations_info['images']))
    print('所有标注的数量：',  len(annotations_info['annotations']))
    print('所有类别的数量：',  len(annotations_info['categories']))


# split
def xml2json_split(data_dir='./data/train', write_dir='./annotations.json', train_p=0.9, val_p=0.1):
    # data_dir = './data/train' #根目录文件，其中包含image文件夹和box文件夹（根据自己的情况修改这个路径）

    image_file_dir = os.path.join(data_dir, 'image')
    xml_file_dir = os.path.join(data_dir, 'box')

    annotations_info = {'images': [], 'annotations': [], 'categories': []}

    categories_map = {"__background__":0, 'holothurian': 1, 'echinus': 2, 'scallop': 3, 'starfish': 4}

    for key in categories_map:
        categoriy_info = {"id":categories_map[key], "name":key}
        annotations_info['categories'].append(categoriy_info)

    train_annotations_info = deepcopy(annotations_info)
    val_annotations_info = deepcopy(annotations_info)



    file_names = [image_file_name.split('.')[0]
                  for image_file_name in os.listdir(image_file_dir)]
    # 去除噪声样本中仅有扇贝的样本
    file_names = filter(lambda file_name:x, file_names)

    # add: get train, val
    file_names_copy = file_names.copy()
    shuffle(file_names_copy)
    train_file_names = file_names_copy[:int(train_p * len(file_names_copy))]
    val_file_names = file_names_copy[int((1-val_p) * len(file_names_copy)):]


    ann_id = 1
    for i, file_name in enumerate(tqdm(file_names)):

        image_file_name = file_name + '.jpg'
        xml_file_name = file_name + '.xml'
        image_file_path = os.path.join(image_file_dir, image_file_name)
        xml_file_path = os.path.join(xml_file_dir, xml_file_name)

        image_info = dict()
        image = cv2.cvtColor(cv2.imread(image_file_path), cv2.COLOR_BGR2RGB)
        height, width, _ = image.shape
        image_info = {'file_name': image_file_name, 'id': i+1,
                      'height': height, 'width': width}
        annotations_info['images'].append(image_info)

        if file_name in train_file_names:
            train_annotations_info['images'].append(image_info)
        if file_name in val_file_names:
            val_annotations_info['images'].append(image_info)

        DOMTree = xml.dom.minidom.parse(xml_file_path)
        collection = DOMTree.documentElement

        names = collection.getElementsByTagName('name')
        names = [name.firstChild.data for name in names]

        xmins = collection.getElementsByTagName('xmin')
        xmins = [xmin.firstChild.data for xmin in xmins]
        ymins = collection.getElementsByTagName('ymin')
        ymins = [ymin.firstChild.data for ymin in ymins]
        xmaxs = collection.getElementsByTagName('xmax')
        xmaxs = [xmax.firstChild.data for xmax in xmaxs]
        ymaxs = collection.getElementsByTagName('ymax')
        ymaxs = [ymax.firstChild.data for ymax in ymaxs]

        object_num = len(names)


        for j in range(object_num):
            if names[j] in categories_map:
                image_id = i + 1
                x1,y1,x2,y2 = int(xmins[j]),int(ymins[j]),int(xmaxs[j]),int(ymaxs[j])
                x1,y1,x2,y2 = x1 - 1,y1 - 1,x2 - 1,y2 - 1

                if x2 == width:
                    x2 -= 1
                if y2 == height:
                    y2 -= 1

                x,y = x1,y1
                w,h = x2 - x1 + 1,y2 - y1 + 1
                category_id = categories_map[names[j]]
                area = w * h
                annotation_info = {"id": ann_id, "image_id":image_id, "bbox":[x, y, w, h], "category_id": category_id, "area": area,"iscrowd": 0}
                annotations_info['annotations'].append(annotation_info)


                if file_name in train_file_names:
                    # print(annotation_info)
                    # print(train_annotations_info)
                    train_annotations_info['annotations'].append(annotation_info)
                    # print(train_annotations_info)
                    # exit()
                if file_name in val_file_names:
                    val_annotations_info['annotations'].append(annotation_info)

                # exit()


                ann_id += 1

    with open(write_dir, 'w') as f:
        json.dump(annotations_info, f, indent=4)

    print('---整理后的标注文件---')
    print('所有图片的数量：',  len(annotations_info['images']))
    print('所有标注的数量：',  len(annotations_info['annotations']))
    print('所有类别的数量：',  len(annotations_info['categories']))

    with open(write_dir.split(".json")[0] + "_train1.json", 'w') as f:
        json.dump(train_annotations_info, f, indent=4)

    print('---训练集---')
    print('训练集所有图片的数量：',  len(train_annotations_info['images']))
    print('训练集所有标注的数量：',  len(train_annotations_info['annotations']))
    print('训练集所有类别的数量：',  len(train_annotations_info['categories']))

    with open(write_dir.split(".json")[0] + "_val1.json", 'w') as f:
        json.dump(val_annotations_info, f, indent=4)

    print('---验证集---')
    print('验证集所有图片的数量：',  len(val_annotations_info['images']))
    print('验证集所有标注的数量：',  len(val_annotations_info['annotations']))
    print('验证集所有类别的数量：',  len(val_annotations_info['categories']))





if __name__ == "__main__":
    # xml2json("/media/gdut502/139e9283-5fa3-498b-ba3d-0272a299eeeb/lrh/和鲸社区水下光学目标检测智能算法赛/dataset/train", "./annotations.json")
    xml2json_split("/media/gdut502/139e9283-5fa3-498b-ba3d-0272a299eeeb/lrh/和鲸社区水下光学目标检测智能算法赛/dataset/train", train_p=0.8, val_p=0.2)
    pass