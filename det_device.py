from ultralytics import YOLO
import torch
import os
import json
from tqdm import tqdm
from PIL import Image, ImageDraw
import utility

def _check_image_file(path):
    img_end = {'jpg', 'bmp', 'png', 'jpeg', 'rgb', 'tif', 'tiff', 'gif', 'pdf'}


    return any([path.lower().endswith(e) for e in img_end])
def get_image_file_list(img_file):
    imgs_lists = []
    if img_file is None or not os.path.exists(img_file):
        raise Exception("not found any img file in {}".format(img_file))

    if os.path.isfile(img_file) and _check_image_file(img_file):
        imgs_lists.append(img_file)
    elif os.path.isdir(img_file):
        for single_file in os.listdir(img_file):
            file_path = os.path.join(img_file, single_file)
            if os.path.isfile(file_path) and _check_image_file(file_path):
                imgs_lists.append(file_path)
    if len(imgs_lists) == 0:
        raise Exception("not found any img file in {}".format(img_file))
    imgs_lists = sorted(imgs_lists)
    return imgs_lists


if __name__ == "__main__":
    args = utility.parse_args()
    model = YOLO(args.yolo_dir)
    path = args.image_dir
    save_path = args.device_det_dir
    save_results = []
    image_file_list = get_image_file_list(path)
    for idx, image_file in tqdm(enumerate(image_file_list)):
        device_results = model(image_file)
        device_boxes = device_results[0].boxes.xywh
        device_boxes_center =  torch.stack([device_boxes[:, 0] + device_boxes[:, 2] / 2, device_boxes[:, 1] + device_boxes[:, 3] / 2], dim = 1).long().cpu().numpy()
        device_class = device_results[0].boxes.cls.long().tolist()
        real_class = [device_results[0].names[i] for i in device_class]
        res = [{
                "device": real_class,
                "device_point": device_boxes.tolist(),
            }]
        save_pred = os.path.basename(image_file) + "\t" + json.dumps(res, ensure_ascii=False) + "\n"
        save_results.append(save_pred)


    print("finish")
    with open(save_path, 'w', encoding='utf-8') as f:
        f.writelines(save_results)


