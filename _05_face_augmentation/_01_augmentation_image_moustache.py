import cv2

img = cv2.imread("data/images/golsa.jpg")

img_moustache_org = cv2.imread("data/images/moustache.png", -1)
moustache_mask_org = img_moustache_org[..., 3]
img_moustache_org = img_moustache_org[..., :3]

# cv2.imshow('mask', moustache_mask_org)
# cv2.imshow('m', img_moustache_org)

face_detector = cv2.CascadeClassifier('data/models/haarcascade_frontalface_default.xml')
nose_detector = cv2.CascadeClassifier('data/models/haarcascade_mcs_nose.xml')

face_boxes = face_detector.detectMultiScale(img)

for face_box in face_boxes:
    fx_1, fy_1, f_w, f_h = face_box
    fx_2, fy_2 = fx_1 + f_w, fy_1 + f_h

    face = img[fy_1:fy_2, fx_1:fx_2]

    # nose section
    nose_boxes = nose_detector.detectMultiScale(face)
    if len(nose_boxes) == 0:
        continue
    else:
        nose_box = nose_boxes[0]

    nx_1, ny_1, n_w, n_h = nose_box
    nx_2, ny_2 = nx_1 + n_w, ny_1 + n_h

    mx_1 = nx_1 - n_w // 2
    mx_2 = nx_2 + n_w // 2

    my_1 = ny_2 - n_h // 3
    my_2 = ny_2 + n_h // 4

    m_w = mx_2 - mx_1
    m_h = my_2 - my_1

    img_moustache = cv2.resize(img_moustache_org, (m_w, m_h))
    mask = cv2.resize(moustache_mask_org, (m_w, m_h))
    inv_mask = cv2.bitwise_not(mask)
    # cv2.imshow("inv_mask",inv_mask)

    face_moustache = face[my_1:my_2, mx_1:mx_2]

    moustache_foreground = cv2.bitwise_and(img_moustache, img_moustache, mask=mask)
    moustache_background = cv2.bitwise_and(face_moustache, face_moustache, mask=inv_mask)

    moustache = cv2.add(moustache_background, moustache_foreground)
    face[my_1:my_2, mx_1:mx_2] = moustache

    # cv2.rectangle(img, (fx_1, fy_1), (fx_2, fy_2), (0, 255, 0), 1)
    # cv2.rectangle(face, (nx_1, ny_1), (nx_2, ny_2), (255, 255, 0), 1)

cv2.imshow('goli', img)
cv2.waitKey(0)
