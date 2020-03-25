import cv2
import mtcnn

face_detector = mtcnn.MTCNN()
img = cv2.imread('friends.jpg')
conf_t = 0.99

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
results = face_detector.detect_faces(img_rgb)

print(results)
for res in results:
    x1, y1, width, height = res['box']
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height

    confidence = res['confidence']
    if confidence < conf_t:
        continue
    key_points = res['keypoints'].values()

    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), thickness=2)
    cv2.putText(img, f'conf: {confidence:.3f}', (x1, y1), cv2.FONT_ITALIC, 1, (0, 0, 255), 1)

    for point in key_points:
        cv2.circle(img, point, 5, (0, 255, 0), thickness=-1)

cv2.imshow('friends', img)
cv2.waitKey(0)
