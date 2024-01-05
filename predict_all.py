import os
import sys
os.environ["FLAGS_allocator_strategy"] = 'auto_growth'

import cv2
import copy
import numpy as np
import json
import time
import logging
from PIL import Image
import utility
import paddleocr.tools.infer.predict_rec as predict_rec
import paddleocr.tools.infer.predict_det as predict_det
import paddleocr.tools.infer.predict_cls as predict_cls
from paddleocr.ppocr.utils.utility import get_image_file_list, check_and_read
from paddleocr.ppocr.utils.logging import get_logger
from utility import draw_ocr_box_txt_, get_rotate_crop_image, get_minarea_rect_crop
logger = get_logger()


class TextSystem(object):
    def __init__(self, args):
        if not args.show_log:
            logger.setLevel(logging.INFO)

        self.text_detector = predict_det.TextDetector(args)
        self.text_recognizer = predict_rec.TextRecognizer(args)
        self.use_angle_cls = args.use_angle_cls
        self.drop_score = args.drop_score
        if self.use_angle_cls:
            self.text_classifier = predict_cls.TextClassifier(args)

        self.args = args
        self.crop_image_res_index = 0

    def draw_crop_rec_res(self, output_dir, img_crop_list, rec_res):
        os.makedirs(output_dir, exist_ok=True)
        bbox_num = len(img_crop_list)
        for bno in range(bbox_num):
            cv2.imwrite(
                os.path.join(output_dir,
                             f"mg_crop_{bno+self.crop_image_res_index}.jpg"),
                img_crop_list[bno])
            logger.debug(f"{bno}, {rec_res[bno]}")
        self.crop_image_res_index += bbox_num

    def __call__(self, img, cls=True):
        time_dict = {'det': 0, 'rec': 0, 'cls': 0, 'all': 0}

        if img is None:
            logger.debug("no valid image provided")
            return None, None, time_dict

        start = time.time()
        ori_im = img.copy()
        dt_boxes, elapse = self.text_detector(img)
        time_dict['det'] = elapse

        if dt_boxes is None:
            logger.debug("no dt_boxes found, elapsed : {}".format(elapse))
            end = time.time()
            time_dict['all'] = end - start
            return None, None, time_dict
        else:
            logger.debug("dt_boxes num : {}, elapsed : {}".format(
                len(dt_boxes), elapse))
        img_crop_list = []

        dt_boxes = sorted_boxes(dt_boxes)

        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno])
            if self.args.det_box_type == "quad":
                img_crop = get_rotate_crop_image(ori_im, tmp_box)
            else:
                img_crop = get_minarea_rect_crop(ori_im, tmp_box)
            img_crop_list.append(img_crop)
        if self.use_angle_cls and cls:
            img_crop_list, angle_list, elapse = self.text_classifier(
                img_crop_list)
            time_dict['cls'] = elapse
            logger.debug("cls num  : {}, elapsed : {}".format(
                len(img_crop_list), elapse))

        rec_res, elapse = self.text_recognizer(img_crop_list)
        time_dict['rec'] = elapse
        logger.debug("rec_res num  : {}, elapsed : {}".format(
            len(rec_res), elapse))
        if self.args.save_crop_res:
            self.draw_crop_rec_res(self.args.crop_res_save_dir, img_crop_list,
                                   rec_res)
        filter_boxes, filter_rec_res = [], []
        for box, rec_result in zip(dt_boxes, rec_res):
            text, score = rec_result
            if score >= self.drop_score:
                filter_boxes.append(box)
                filter_rec_res.append(rec_result)
        end = time.time()
        time_dict['all'] = end - start
        return filter_boxes, filter_rec_res, time_dict


def sorted_boxes(dt_boxes):
    """
    Sort text boxes in order from top to bottom, left to right
    args:
        dt_boxes(array):detected text boxes with shape [4, 2]
    return:
        sorted boxes(array) with shape [4, 2]
    """
    num_boxes = dt_boxes.shape[0]
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)

    for i in range(num_boxes - 1):
        for j in range(i, -1, -1):
            if abs(_boxes[j + 1][0][1] - _boxes[j][0][1]) < 10 and \
                    (_boxes[j + 1][0][0] < _boxes[j][0][0]):
                tmp = _boxes[j]
                _boxes[j] = _boxes[j + 1]
                _boxes[j + 1] = tmp
            else:
                break
    return _boxes

def match(points1, points2):

    distances = np.linalg.norm(points1[:, np.newaxis] - points2, axis=2)
    # 获取距离从小到大排序的索引
    indices = np.unravel_index(np.argsort(distances, axis=None), distances.shape)

    # 用于存储已经取走的点
    taken_points_x = set()
    taken_points_y = set()
    matched = []
    # 逐步取走距离最小的两个点
    for i, j in zip(*indices):
        if i not in taken_points_x and j not in taken_points_y:
            matched.append([i, j])
            # print(f"Take points: {i, j}, {points1[i]}, {points2[j]}, Distance: {distances[i, j]}")
            taken_points_x.add(i)
            taken_points_y.add(j)
    left_1 = set(range(points1.shape[0])) - taken_points_x
    left_2 = set(range(points2.shape[0])) - taken_points_y

    return matched, left_1, left_2


def main(args):
    image_file_list = get_image_file_list(args.image_dir)
    image_file_list = image_file_list[args.process_id::args.total_process_num]
    text_sys = TextSystem(args)
    is_visualize = True
    font_path = args.vis_font_path
    drop_score = args.drop_score
    draw_img_save_dir = args.draw_img_save_dir
    os.makedirs(draw_img_save_dir, exist_ok=True)
    save_results = []

    total_time = 0
    _st = time.time()
    path = args.device_det_dir
    f = open(path, "r", encoding='utf-8')
    names = {'安防1':0, '安防2':1, '箱柜':2, 'cabinet':3, 'UFO': 4}
    all_device = f.readlines()
    for idx, image_file in enumerate(image_file_list):
        img_name, device = all_device[idx].split("\t")
        if img_name != os.path.basename(image_file):
            logger.debug("file name mismatch! {}, {}".format(img_name, image_file))
            continue
        img, flag_gif, flag_pdf = check_and_read(image_file)
        if not flag_gif and not flag_pdf:
            img = cv2.imread(image_file)
        if not flag_pdf:
            if img is None:
                logger.debug("error in loading image:{}".format(image_file))
                continue
            imgs = [img]
        else:
            page_num = args.page_num
            if page_num > len(img) or page_num == 0:
                page_num = len(img)
            imgs = img[:page_num]

        # start
        for index, img in enumerate(imgs):
            starttime = time.time()

            device = json.loads(device)[0]
            
            real_class, device_points = device['device'], device["device_point"]
            device_points = np.array(device_points)
            dt_boxes, rec_res, time_dict = text_sys(img)
            tag1, tag2 = False, False
            if device_points.size == 0:
                tag1 = True
            if len(dt_boxes) == 0:
                tag2 = True
            save_name = os.path.join(draw_img_save_dir, os.path.basename(image_file))

            if tag1 and tag2:
                logger.debug(f"detect no device and no text, file_path is {image_file}")
                save_results.append(os.path.basename(image_file) + "\t[]\n")
                cv2.imwrite(save_name, img)
                logger.debug("The visualized image saved in {}".format(save_name))

            elif tag1:
                logger.debug(f"detect no device, file_path is {image_file}")
                res = [{
                    "transcription": rec_res[j][0],
                    "text_pos": dt_boxes_center[j].tolist(),
                } for j in range(len(dt_boxes))]
                save_pred = os.path.basename(image_file) + "\t" + json.dumps(
                    res, ensure_ascii=False)
                save_results.append(save_pred)
                if is_visualize:

                    image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    boxes = dt_boxes
                    txts = [rec_res[i][0] for i in range(len(rec_res))]
                    scores = [rec_res[i][1] for i in range(len(rec_res))]

                    draw_img = draw_ocr_box_txt_(
                        image,
                        tag1=True,
                        boxes=boxes,
                        txts=txts,
                        scores=scores,
                        drop_score=drop_score,
                        font_path=font_path)

                    cv2.imwrite(save_name, draw_img[:, :, ::-1])
                    logger.debug("The visualized image saved in {}".format(save_name))

            elif tag2:
                logger.debug(f"detect no text, file_path is {image_file}")
                res = [{
                    "device": real_class[i],
                    "device_pos": device_points[i].tolist(),
                } for i in range(len(real_class))]
                save_pred = os.path.basename(image_file) + "\t" + json.dumps(
                    res, ensure_ascii=False)
                save_results.append(save_pred)

                if is_visualize:
                    x, y, w, h = device_points[:, 0], device_points[:, 1], device_points[:, 2], device_points[:, 3]
                    x1, y1 = x - w / 2, y - h / 2
                    x2, y2 = x1 + w, y1 + h
                    new_boxes = np.stack([x1, y1, x2, y1, x2, y2, x1, y2], axis = 1).reshape(-1, 4, 2)
                    image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

                    draw_img = draw_ocr_box_txt_(
                        image,
                        tag2=True,
                        device=new_boxes,
                        names=real_class,
                        cls=names,
                        drop_score=drop_score,
                        font_path=font_path)

                    cv2.imwrite(save_name, draw_img[:, :, ::-1])
                    logger.debug("The visualized image saved in {}".format(save_name))

            else:

                device_boxes_center = np.stack([device_points[:, 0], device_points[:, 1]], axis=1)
            
                dt_boxes_center = np.array([[(box[0][0] + box[2][0]) / 2, (box[0][1] + box[2][1]) / 2 ] for box in dt_boxes])
                matched, left_device, left_text = match(device_boxes_center, dt_boxes_center)
                elapse = time.time() - starttime
                total_time += elapse
                if len(imgs) > 1:
                    logger.debug(
                        str(idx) + '_' + str(index) + "  Predict time of %s: %.3fs"
                        % (image_file, elapse))
                else:
                    logger.debug(
                        str(idx) + "  Predict time of %s: %.3fs" % (image_file, elapse))

                res = [{
                        "transcription": rec_res[j][0],
                        "text_pos": dt_boxes[j].tolist(),
                        "device": real_class[i],
                        "device_pos": device_points[i].tolist(),
                } for i, j in matched]
                left = ""
                if left_text:
                    left = [{
                        "transcription": rec_res[j][0],
                        "text_pos": dt_boxes_center[j].tolist(),
                    } for j in left_text]
                if left_device:
                    left = [{
                        "device": real_class[i],
                        "device_pos": device_boxes_center[i].tolist(),
                    } for i in left_device]

                if len(imgs) > 1:
                    save_pred = os.path.basename(image_file) + '_' + str(
                        index) + "\t" + json.dumps(
                            res, ensure_ascii=False)
                    if left != "":
                        save_pred += " " + json.dumps(left, ensure_ascii=False) 
                else:
                    save_pred = os.path.basename(image_file) + "\t" + json.dumps(
                        res, ensure_ascii=False)
                    if left != "":
                        save_pred += " " + json.dumps(left, ensure_ascii=False)
                save_pred += "\n"
                save_results.append(save_pred)

                if is_visualize:
                    x, y, w, h = device_points[:, 0], device_points[:, 1], device_points[:, 2], device_points[:, 3]
                    x1, y1 = x - w / 2, y - h / 2
                    x2, y2 = x1 + w, y1 + h
                    new_boxes = np.stack([x1, y1, x2, y1, x2, y2, x1, y2], axis = 1).reshape(-1, 4, 2)
                    image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    boxes = dt_boxes
                    txts = [rec_res[i][0] for i in range(len(rec_res))]
                    scores = [rec_res[i][1] for i in range(len(rec_res))]

                    draw_img = draw_ocr_box_txt_(
                        image,
                        device=new_boxes,
                        names=real_class,
                        cls=names,
                        boxes=boxes,
                        txts=txts,
                        scores=scores,
                        drop_score=drop_score,
                        font_path=font_path)

                    cv2.imwrite(save_name, draw_img[:, :, ::-1])
                    logger.debug("The visualized image saved in {}".format(save_name))

    logger.info("The predict total time is {}".format(time.time() - _st))
    if args.benchmark:
        text_sys.text_detector.autolog.report()
        text_sys.text_recognizer.autolog.report()

    with open(
            os.path.join(draw_img_save_dir, "system_results.txt"),
            'w',
            encoding='utf-8') as f:
        f.writelines(save_results)


if __name__ == "__main__":
    args = utility.parse_args()
    main(args)