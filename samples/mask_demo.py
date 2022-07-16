import cv2
import numpy as np


def create_mask(img):

    # 多边形 mask
    # 先创建个和输入图片相同大小的 zero 图
    zero_img = np.zeros(img.shape[:2], dtype="uint8")
    # mask = cv2.polylines(mask_img, b, 1, 255)
    # 想要的掩码图区域
    pts = np.array([[(22,59),(54,24),(106,9),(200,14),(223,39),(228,75),(277,82),(303,108),(317,133),(302,153),(234,157),(151,157),(55,157)]], dtype=np.int32)

    # zero 图上用白色填充 mask
    mask = cv2.fillPoly(zero_img, pts, color=(255, 255, 255))
    print(mask.shape)
    # 圆形 mask
    x = 140
    y = 100
    r = 80
    # mask_img = np.zeros(img.shape[:2], dtype=np.uint8)
    # mask = cv2.circle(mask_img, (x, y), r, (255, 255, 255), -1)  # 在掩码图上画个圈, x, y 指定位置，填充白色.

    # 返回
    return mask


if __name__ == "__main__":
    """
    背景画掩码
    
    """

    img = cv2.imread("test_image.jpeg")
    dst = cv2.imread("test_image.jpeg")
    mask = create_mask(img)
    # 输入图片和掩码图和在一起，只有 mask 区域漏了出来，0 的地方都是黑的
    image = cv2.add(img, np.zeros(np.shape(img), dtype=np.uint8), mask=mask)  # 输入网络
    print(mask.shape, image.shape)
    fake_image = image  # 网络得到结果

    cv2.imshow("img", img)
    # 展示掩膜图片
    cv2.imshow("mask", mask)
    # 展示添加掩膜效果图片
    cv2.imshow("image", image)

    background = img - fake_image   # 拿到一张和掩码图 0 的位置相反的图片， 也就是说只有前景位置像素点是 0

    res = background + fake_image   # 前景位置为 0 的加上 背景位置为 0 的，颜色都显示出来噜，而且严丝合缝 so nice.
    print("res", img.shape, fake_image.shape)

    # pts = np.array([[(22,59),(54,24),(106,9),(200,14),(223,39),(228,75),(277,82),(303,108),(317,133),(302,153),(234,157),(151,157),(55,157)]], dtype=np.int32)
    cv2.imshow("background", background)
    cv2.imshow("res", res)

    # mask = cv2.fillPoly(img, pts, color=(255, 255, 255))

    # output1 = cv2.seamlessClone(img, dst, mask, (180, 108), cv2.MIXED_CLONE)
    # cv2.imshow("pos", output1)

    while True:
        try:
            cv2.waitKey(0)
        except Exception as e:
            break

    cv2.waitKey(0)
    cv2.destroyAllWindow()