import numpy as np
import cv2
from scipy.spatial.distance import euclidean
import mtcnn


def align(img, left_eye_pos, right_eye_pos, size=(112, 112), eye_pos=(0.35, 0.4)):
    width, height = size
    eye_pos_w, eye_pos_h = eye_pos

    l_e, r_e = left_eye_pos, right_eye_pos

    dy = r_e[1] - l_e[1]
    dx = r_e[0] - l_e[0]
    dist = euclidean(l_e, r_e)
    scale = (width * (1 - 2 * eye_pos_w)) / dist

    # get rotation
    center = ((l_e[0] + r_e[0]) // 2, (l_e[1] + r_e[1]) // 2)
    angle = np.degrees(np.arctan2(dy, dx)) + 360

    m = cv2.getRotationMatrix2D(center, angle, scale)
    tx = width * 0.5
    ty = height * eye_pos_h
    m[0, 2] += (tx - center[0])
    m[1, 2] += (ty - center[1])

    aligned_face = cv2.warpAffine(img, m, (width, height))
    return aligned_face


if __name__ == '__main__':
    img = cv2.imread('pooya.jpg')
    face_detector = mtcnn.MTCNN()
    results = face_detector.detect_faces(img)
    l_e = results[0]['keypoints']['left_eye']
    r_e = results[0]['keypoints']['right_eye']

    aligned_face = align(img, left_eye_pos=l_e, right_eye_pos=r_e)
    cv2.imshow('', aligned_face)
    cv2.imshow('pooya', img)
    cv2.waitKey(0)
