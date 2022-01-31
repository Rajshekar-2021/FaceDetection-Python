import cv2
import mediapipe as mp
import time


class FaceDetector():

    def __init__(self):
        self.mpFaceDetection = mp.solutions.face_detection
        self.face = self.mpFaceDetection.FaceDetection()
        self.mpDraw = mp.solutions.drawing_utils

    def findFaces(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face.process(imgRGB)
        # print(results.detections)

        return img

    def findFacePosition(self, img, draw=True):

        bboxList = []
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face.process(imgRGB)

        if results.detections:
            for id, detection in enumerate(results.detections):
                bboxC = detection.location_data.relative_bounding_box

                ih, iw, ic = img.shape

                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                       int(bboxC.width * iw), int(bboxC.height * ih)

                bboxList.append([id, bbox, detection.score])

                cv2.rectangle(img, bbox, (255, 0, 255), 2)
                cv2.putText(img, f'{int(detection.score[0] * 100)}%',
                            (bbox[0], bbox[1] - 20), cv2.QT_FONT_NORMAL, 1, (0, 255, 0))

                # print(bboxList)
                return bboxList


def main():
    cap = cv2.VideoCapture('Videos/6.mp4')
    pTime = 0

    detector = FaceDetector()

    while True:
        success, img = cap.read()
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        bboxList = detector.findFacePosition(img)
        img = detector.findFaces(img)


        print(bboxList)

        cv2.putText(img, f'FPS : {int(fps)}%', (5, 30), cv2.QT_FONT_NORMAL, 1, (0, 255, 0))
        cv2.imshow("Video", img)
        cv2.waitKey(10)


if __name__ == "__main__":
    main()
