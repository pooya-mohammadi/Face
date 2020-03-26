import numpy as np
import cv2
from scipy.spatial.distance import euclidean
import mtcnn

eye_pos_w, eye_pos_h = 0.35, 0.4
width, height = 112, 112

img = cv2.imread('pooya.jpg')
h, w, _ = img.shape

img_rgb = img[..., ::-1]
face_detector = mtcnn.MTCNN()
results = face_detector.detect_faces(img_rgb)

l_e = results[0]['keypoints']['left_eye']
r_e = results[0]['keypoints']['right_eye']
center = (((r_e[0] + l_e[0]) // 2), ((r_e[1] + l_e[1]) // 2))

dx = (r_e[0] - l_e[0])
dy = (r_e[1] - l_e[1])
dist = euclidean(l_e, r_e)

angle = np.degrees(np.arctan2(dy, dx)) + 360
scale = width * (1 - (2 * eye_pos_w)) / dist

tx = width * 0.5
ty = height * eye_pos_h

m = cv2.getRotationMatrix2D(center, angle, scale)

m[0, 2] += (tx - center[0])
m[1, 2] += (ty - center[1])


face_align = cv2.warpAffine(img, m, (width, height))

cv2.imshow('align', face_align)
cv2.imshow('org', img)
cv2.waitKey(0)
