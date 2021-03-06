import cv2
import numpy as np


path_to_frozen_inference_graph = 'frozen_inference_graph_coco.pb'
path_coco_model= 'mask_rcnn_inception_v2_coco_2018_01_28.pbtxt'
net = cv2.dnn.readNetFromTensorflow(path_to_frozen_inference_graph, path_coco_model)
image_path = 'test_image.jpeg'
img = cv2.imread(image_path)
ori_img = img
height, width, _ = img.shape

colors = np.random.randint(125, 255, (80, 3))

img = cv2.resize(img, (650, 550))

# Generate balck image
black_image = np.zeros((height, width, 3), np.uint8)
black_image[:] = (0, 0, 0)
blob = cv2.dnn.blobFromImage(img, swapRB=True)

net.setInput(blob)
boxes, masks = net.forward(["detection_out_final", "detection_masks"])
print(f"Mask shape: {masks.shape} M: {masks}\n\n")

detection_count = boxes.shape[2]
print(detection_count)
# print(boxes, masks)

for i in range(detection_count):
     box = boxes[0, 0, i]
     class_id, score = box[1], box[2]

     print(class_id, box)

     if score < 0.5:
         continue

     x, y = int(box[3] * width),  int(box[4] * height)
     x2, y2 = int(box[5] * width), int(box[6] * height)

     roi = black_image[y: y2, x: x2]
     roi_height, roi_width, _ = roi.shape

     print(roi_height, roi_width)  # 363 646

     mask = masks[i, int(class_id)]     # 15,15

     # mask 和 threshold 大小一样
     mask = cv2.resize(mask, (roi_width, roi_height))
     cv2.imwrite("mask.jpg", mask)

     _, mask = cv2.threshold(mask, 0.5, 255, cv2.THRESH_BINARY)
     cv2.imwrite("threshold.jpg", mask)

     # cv2.rectangle(img, (x, y), (x2, y2), (255, 0, 0), 3)
     # 查找轮廓
     contours, _ = cv2.findContours(np.array(mask, np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

     color = colors[int(class_id)]

     for cnt in contours:
          print(f"pts: {cnt} \n")
          mask1 = cv2.fillPoly(roi, [cnt], (int(color[0]), int(color[1]), int(color[2])))
          print("mask", mask1.shape)
          cv2.imwrite("m1.jpg", mask1)
          mm = mask1.resize(height, width)
          t = cv2.add(ori_img, np.zeros(np.shape(ori_img.resize()), dtype=np.uint8), mask=mm)  # 输入网络
          # cv2.imwrite("t.jpg", t)

# cv2.imshow("Final",np.hstack([img,black_image]))
# cv2.imshow("Overlay_image",((0.6*black_image)+(0.4*img)).astype("uint8"))
# cv2.waitKey(0)
