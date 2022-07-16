import face_recognition
import cv2
import numpy as np

# Open Eye Images
eye = cv2.imread('eye.png')

# Open Face image
face = cv2.imread('face.jpeg')
# Get Keypoints
image = face_recognition.load_image_file('face.jpeg')
face_landmarks_list = face_recognition.face_landmarks(image)
for facemarks in face_landmarks_list:
    # Get Eye Data
    eyeLPoints = facemarks['left_eye']
    eyeRPoints = facemarks['right_eye']
    npEyeL = np.array(eyeLPoints)

print(eyeLPoints)
# These points define the contour of the eye in the EYE image
# point_list = [(51, 228), (100, 151), (233, 102), (338, 110), (426, 160), (373, 252), (246, 284), (134, 268)]
# point_list = [(244, 613), (289, 587), (339, 587), (386, 621), (338, 629), (287, 630),  (246, 284), (134, 268)]
# point_list = [(10, 20), (30, 40), (5, 6)]
point_list = [(51, 228), (100, 151), (233, 102), (338, 110), (426, 160), (373, 252), (246, 284), (134, 268)]

poly_left = np.array(point_list, np.int32)

# point_size = 6
# point_color = (0, 255, 255) # BGR
# thickness = 4 # 可以为 0 、4、8
# for point in point_list:
#     p1 = cv2.circle(image, point, point_size, point_color, thickness)
#     # p2 = cv2.circle(eye, point, point_size, point_color, thickness)
#     cv2.imwrite('p1.png', p1)
#     # cv2.imwrite('p2.png', p2)


# Create a mask for the eye
src_mask = np.zeros(face.shape, face.dtype)
cv2.fillPoly(src_mask, [poly_left], (255, 255, 255))
cv2.imwrite('src_mask.png', src_mask)

# Find where the eye should go
center, r = cv2.minEnclosingCircle(npEyeL)
center = tuple(np.array(center, int))

# Clone seamlessly.
output = cv2.seamlessClone(eye, face, src_mask, center, cv2.NORMAL_CLONE)
cv2.imwrite("a.png", output)