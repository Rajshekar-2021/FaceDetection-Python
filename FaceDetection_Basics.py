import cv2
import mediapipe as mp
import time

#cap = cv2.VideoCapture('Videos/3.mp4')
cap = cv2.VideoCapture(0)

pTime = 0
mpFaceDetection = mp.solutions.face_detection
face = mpFaceDetection.FaceDetection()

mpDraw = mp.solutions.drawing_utils

while True:
    success, img = cap.read()
    img = cv2.flip(img,1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face.process(imgRGB)
    # print(results.detections)

    if results.detections:
        for id, detection in enumerate(results.detections):
            # print(id, detection.location_data.relative_bounding_box)
            # mpDraw.draw_detection(img,detection)
            bboxC = detection.location_data.relative_bounding_box

            ih, iw, ic = img.shape

            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                   int(bboxC.width * iw), int(bboxC.height * ih)
            cv2.rectangle(img,bbox,(255,0,255),2)
            cv2.putText(img, f'{int(detection.score[0]*100)}%',
                        (bbox[0], bbox[1]-20), cv2.QT_FONT_NORMAL, 1, (0, 255, 0))

            print(id,bbox)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS : {int(fps)}%', (5, 30), cv2.QT_FONT_NORMAL, 1, (255, 0, 0))
    cv2.imshow("Video", img)
    cv2.waitKey(1)
