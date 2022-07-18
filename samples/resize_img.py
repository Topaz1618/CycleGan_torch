import os
import cv2

all_image = os.listdir("test_images")

print(all_image)

for img_name in all_image:
    im = cv2.imread(os.path.join("test_images", img_name))
    print(im)
    res = cv2.resize(im, [256, 256])
    cv2.imwrite(os.path.join("test_images", f"t{img_name}"), res)