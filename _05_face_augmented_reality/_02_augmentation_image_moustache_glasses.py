import cv2

img = cv2.imread("data/images/golsa.jpg")

img_moustache_org = cv2.imread("data/images/moustache.png", -1)
moustache_mask_org = img_moustache_org[..., 3]
img_moustache_org = img_moustache_org[..., :3]

img_glasses_org = cv2.imread("data/images/glasses.png", -1)
glasses_mask_org = img_glasses_org[..., 3]
img_glasses_org = img_glasses_org[..., :3]

face_detector = cv2.CascadeClassifier('data/models/haarcascade_frontalface_default.xml')
nose_detector = cv2.CascadeClassifier('data/models/haarcascade_mcs_nose.xml')
eye_detector = cv2.CascadeClassifier('data/models/haarcascade_mcs_eyepair_big.xml')

face_boxes = face_detector.detectMultiScale(img)

for face_box in face_boxes:
    fx_1, fy_1, f_w, f_h = face_box
    fx_2, fy_2 = fx_1 + f_w, fy_1 + f_h
    face = img[fy_1:fy_2, fx_1:fx_2]

    # nose section
    nose_box = nose_detector.detectMultiScale(face)
    if len(nose_box) == 0:
        continue
    else:
        nose_box = nose_box[0]
    nx_1, ny_1, nw, nh = nose_box
    nx_2, ny_2 = nx_1 + nw, ny_1 + nh

    # moustache section
    mx_1 = nx_1 - nw // 2
    mx_2 = nx_2 + nw // 2

    my_1 = ny_2 - nh // 3
    my_2 = ny_2 + nh // 4

    my_2 = f_h if my_2 > f_h else my_2
    my_1 = 0 if my_1 < 0 else my_1
    mx_2 = f_w if mx_2 > f_w else mx_2
    mx_1 = 0 if mx_1 < 0 else mx_1

    mw = mx_2 - mx_1
    mh = my_2 - my_1

    img_moustache = cv2.resize(img_moustache_org, (mw, mh))
    moustache_mask = cv2.resize(moustache_mask_org, (mw, mh))

    inv_moustache_mask = cv2.bitwise_not(moustache_mask)
    face_moustache = face[my_1:my_2, mx_1:mx_2]

    background_moustache = cv2.bitwise_and(face_moustache, face_moustache, mask=inv_moustache_mask)
    foreground_moustache = cv2.bitwise_and(img_moustache, img_moustache, mask=moustache_mask)

    moustache = cv2.add(background_moustache, foreground_moustache)

    face[my_1:my_2, mx_1:mx_2] = moustache

    # eye section
    eye_box = eye_detector.detectMultiScale(face)
    if len(eye_box) == 0:
        continue
    else:
        eye_box = eye_box[0]
    ex_1, ey_1, ew, eh = eye_box
    ex_2, ey_2 = ex_1 + ew, ey_1 + eh

    # glasses section
    gx_1 = ex_1 - ew // 8
    gx_2 = ex_2 + ew // 8

    gy_1 = ey_1
    gy_2 = ey_2

    gy_2 = f_h if gy_2 > f_h else gy_2
    gy_1 = 0 if gy_1 < 0 else gy_1
    gx_2 = f_w if gx_2 > f_w else gx_2
    gx_1 = 0 if gx_1 < 0 else gx_1

    gw = gx_2 - gx_1
    gh = gy_2 - gy_1

    img_glasses = cv2.resize(img_glasses_org, (gw, gh))
    glasses_mask = cv2.resize(glasses_mask_org, (gw, gh))

    inv_glasses_mask = cv2.bitwise_not(glasses_mask)
    face_glasses = face[gy_1:gy_2, gx_1:gx_2]

    background_glasses = cv2.bitwise_and(face_glasses, face_glasses, mask=inv_glasses_mask)
    foreground_glasses = cv2.bitwise_and(img_glasses, img_glasses, mask=glasses_mask)

    glasses = cv2.add(background_glasses, foreground_glasses)
    face[gy_1:gy_2, gx_1:gx_2] = glasses
    break

cv2.imshow('goli', img)
cv2.imwrite('data/results/goli_augmented.jpg', img)
cv2.waitKey(0)
