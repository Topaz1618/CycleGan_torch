"""
Note:
    ori_image

    1. ori image => gan => fake_image
    2. get_mask(fake_image) => fake image front
    3. ori_image - front => background
    4. fake_img front + background = full image

"""


import cv2
import numpy as np
from area import polygon_area

path_to_frozen_inference_graph = '../data/frozen_inference_graph_coco.pb'
path_coco_model = '../data/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt'

net = cv2.dnn.readNetFromTensorflow(path_to_frozen_inference_graph, path_coco_model)
colors = np.random.randint(125, 255, (80, 3))


def get_mask(img_name):
    img = cv2.imread(img_name)
    height, width, _ = img.shape
    black_image = np.zeros(img.shape[:2], dtype="uint8")

    blob = cv2.dnn.blobFromImage(img, swapRB=True)
    net.setInput(blob)
    boxes, masks = net.forward(["detection_out_final", "detection_masks"])
    detection_count = boxes.shape[2]

    LABELS = open("../data/object_detection_classes_coco.txt").read().strip().split("\n")

    for i in range(detection_count):
        box = boxes[0, 0, i]
        class_id = box[1]

        if LABELS[int(class_id)] != "horse":
            continue

        score = box[2]
        if score < 0.5:
            continue

        x = int(box[3] * width)
        y = int(box[4] * height)
        x2 = int(box[5] * width)
        y2 = int(box[6] * height)

        roi = black_image[y: y2, x: x2]
        roi_height, roi_width = roi.shape
        mask = masks[i, int(class_id)]
        mask = cv2.resize(mask, (roi_width, roi_height))
        _, mask = cv2.threshold(mask, 0.5, 255, cv2.THRESH_BINARY)
        # cv2.rectangle(img, (x, y), (x2, y2), (255, 0, 0), 3)

        contours, _ = cv2.findContours(np.array(mask, np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        color = colors[int(class_id)]
        for cnt in contours:
            # cv2.cv2fillPoly(roi, [cnt], (int(color[0]), int(color[1]), int(color[2])))
            cv2.fillPoly(roi, [cnt], color=(255, 255, 255))     # 指定白色
            # cv2.imshow("fake_mask.png", mask_res)
        return black_image


if __name__ == "__main__":
    # todo: 加 for loop
    for i in range(7):
        if i == 2:
            continue
        img_name = f"results/horse{i}_real.png"
        fake_name = f"results/horse{i}_fake.png"
        real_img = cv2.imread(img_name)
        fake_img = cv2.imread(fake_name)
        print(img_name, fake_name)
        # cv2.imshow("Results", real_img)
        # cv2.imshow("Results", fake_img)
        mask_res = get_mask(img_name)

        mask_image = cv2.add(real_img, np.zeros(np.shape(real_img), dtype=np.uint8), mask=mask_res)  # 输入网络
        fake_front = cv2.add(fake_img, np.zeros(np.shape(fake_img), dtype=np.uint8), mask=mask_res)  # 输入网络

        real_background = real_img - mask_image

        res = fake_front + real_background

        cv2.imwrite(f"results/horse_res{i}.png", res)

    # cv2.imshow("mask image", mask_image)
    # cv2.imshow("Real background", real_background)
    # cv2.imshow("fake front", fake_use)
    # cv2.imshow("Results", res)
    # key = cv2.waitKey(0)



