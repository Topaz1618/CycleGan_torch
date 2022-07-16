import numpy as np
import cv2


def change_background(img):
    rgb = np.random.randint(low=0, high=255, size=(3,))
    res = np.tile(img[:, :, None], (1, 1, 3))
    for i in range(3):
        res[:, :, i][res[:, :, i] < 127.5] = rgb[i]
        print(res.shape, type(res))
    return res


def change_numeral(img):
    rgb = np.random.randint(low=0, high=255, size=(3,))
    res = np.tile(img[:, :, None], (1, 1, 3))
    for i in range(3):
        res[:, :, i][res[:, :, i] >= 127.5] = rgb[i]
        res[:, :, i][res[:, :, i] < 127.5] = 255
    print(res.shape, type(res))

    return res


data = np.load("/Users/Topaz/Projects/py_cyclegan/tests/data/mnist/mnist.npz")

for idx, img in enumerate(data['x_train']):
    if idx < 2000:
        # print(idx)
        # print(img.shape, type(img))
        # img = change_background(img)
        img = change_numeral(img)
        cv2.imwrite(f"/Users/topaz/Projects/py_cyclegan/data/mnist/trainB/{idx}.jpg", img)


# colored_background_data = {
#     'x_train': np.concatenate([change_background(img)[None, :, :, :] for img in data['x_train']], 0),
#     'y_train': data['y_train'],
#     'x_test': np.concatenate([change_background(img)[None, :, :, :] for img in data['x_test']], 0),
#     'y_test': data['y_test']
# }
# np.savez("data/mnist/mnist_colorback.npz", **colored_background_data)
#
# colored_numeral_data = {
#     'x_train': np.concatenate([change_numeral(img)[None, :, :, :] for img in data['x_train']], 0),
#     'y_train': data['y_train'],
#     'x_test': np.concatenate([change_numeral(img)[None, :, :, :] for img in data['x_test']], 0),
#     'y_test': data['y_test']
# }
# np.savez("data/mnist/mnist_colornum.npz", **colored_numeral_data)