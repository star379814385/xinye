#!/bin/bash
basepath=$(cd `dirname $0`; pwd)
cd ${basepath}
cd ..

# step1. crop image (train data) and train retrieval model
#python ./utils/data/image_crop.py train a
#python ./utils/data/image_crop.py train b

# step2. train retrieval model using crop image
cd tools
#python ./retrieval_train.py
cd ..

# step3. gen json for object detectin
#python utils/data/split_json_ab1.py

# step4. train object detection
./tools/dist_train.sh ./model/detection/configs/cascade_mask_rcnn_swin_base_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py 1 \
--work-dir ./model_files/detection/cascade_mask_rcnn_swin_base_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco2_1

