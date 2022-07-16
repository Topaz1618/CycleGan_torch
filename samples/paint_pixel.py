import cv2
import numpy as np


def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "(%d,%d)" % (x, y)
        print(xy)
        cv2.circle(img, (x, y), 1, (255, 0, 0), thickness=-1)
        cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), thickness=2)
        cv2.imshow("MyBoard", img)


if __name__ == "__main__":
    img = cv2.imread("cat.jpeg")
    cv2.namedWindow("MyBoard", cv2.WINDOW_KEEPRATIO)
    cv2.setMouseCallback("MyBoard", on_EVENT_LBUTTONDOWN)
    cv2.imshow("MyBoard", img)

    while True:
        try:
            cv2.waitKey(0)
        except Exception as e:
            cv2.destroyWindow("MyBoard")
            break

    cv2.waitKey(0)
    cv2.destroyAllWindow()