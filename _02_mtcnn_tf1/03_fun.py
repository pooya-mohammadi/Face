import mtcnn
import cv2

face_detector = mtcnn.MTCNN(min_face_size=50)
conf_t = 0.99
vc = cv2.VideoCapture(1)

while vc.isOpened():
    ret, frame = vc.read()
    if not ret:
        print(":(")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detector.detect_faces(frame_rgb)

    for res in results:
        x1, y1, width, height = res['box']
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height

        confidence = res['confidence']
        if confidence < conf_t:
            continue

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 100, 100), thickness=2)
        left_eye = res['keypoints']['left_eye']
        right_eye = res['keypoints']['right_eye']
        mouth_left = res['keypoints']['mouth_left']
        mouth_right = res['keypoints']['mouth_right']
        nose = res['keypoints']['nose']

        for eye in [left_eye, right_eye]:
            cv2.rectangle(frame, (eye[0] - 15, eye[1] - 15), (eye[0] + 15, eye[1] + 15), (0, 100, 100), 2)
        cv2.circle(frame, nose, 10, (0, 100, 100), thickness=-1)

        cv2.rectangle(frame, (mouth_left[0] - 15, mouth_left[1] - 15), (mouth_right[0] + 15, mouth_right[1] + 15), (0, 100, 100), 2)

    cv2.imshow('pooya on camera', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
