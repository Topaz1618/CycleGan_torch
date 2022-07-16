"""
Instance segmentation

"""
import cv2
import numpy as np
from area import polygon_area

path_to_frozen_inference_graph = 'data/frozen_inference_graph_coco.pb'
path_coco_model = 'data/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt'

net = cv2.dnn.readNetFromTensorflow(path_to_frozen_inference_graph, path_coco_model)
colors = np.random.randint(125, 255, (80, 3))

img = frame = cv2.imread("test_image_real.png")
height, width, _ = img.shape
# black_image = np.zeros((height, width, 3), np.uint8)
# black_image[:] = (0, 0, 0)

black_image = np.zeros(img.shape[:2], dtype="uint8")


blob = cv2.dnn.blobFromImage(img, swapRB=True)
net.setInput(blob)
boxes, masks = net.forward(["detection_out_final", "detection_masks"])
detection_count = boxes.shape[2]


LABELS = open("data/object_detection_classes_coco.txt").read().strip().split("\n")

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
    # roi_height, roi_width, _ = roi.shape
    roi_height, roi_width = roi.shape
    mask = masks[i, int(class_id)]
    mask = cv2.resize(mask, (roi_width, roi_height))
    _, mask = cv2.threshold(mask, 0.5, 255, cv2.THRESH_BINARY)
    # cv2.rectangle(img, (x, y), (x2, y2), (255, 0, 0), 3)

    contours, _ = cv2.findContours(np.array(mask, np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    color = colors[int(class_id)]
    for cnt in contours:
        cv2.fillPoly(roi, [cnt], (int(color[0]), int(color[1]), int(color[2])))
        my_mask = cv2.fillPoly(roi, [cnt], color=(255, 255, 255)) # 指定白色

# print(my_mask.shape, img.shape, black_image.shape)
image = cv2.add(frame, np.zeros(np.shape(frame), dtype=np.uint8), mask=black_image)  # 输入网络
cv2.imwrite("gan_input1.png", image)
background = img - image
print(image)
# # 展示添加掩膜效果图片
cv2.imshow("what I want", image)
cv2.imshow("background", background)
cv2.imshow("Black image", black_image)

res = background + image
cv2.imshow("res", res)
# (550, 650, 3)(348, 491, 3)
print(black_image.shape, frame.shape)
# final_frame = ((0.6*black_image)+(0.4*frame)).astype("uint8")
# cv2.imshow("Overlay Frames", final_frame)

key = cv2.waitKey(0)


