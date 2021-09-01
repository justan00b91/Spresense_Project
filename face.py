import cv2
import mediapipe as mp
import face_recognition
import time

cap = cv2.VideoCapture(0)
pTime = 0

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection()

while True:
    #face detection

    sucess, img = cap.read()

    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)

    if results.detections:
        for id, detection in enumerate(results.detections):
            mpDraw.draw_detection(imgRGB,detection)

            #face recognition
                
            imageDB = face_recognition.load_image_file("example.jpg")
            #replace example.jpg with a photo name
            face_locations = face_recognition.face_locations(imageDB)
            val = 1

    #debug<>
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(imgRGB,f'FPS: {int(fps)}', (20, 110), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
    cv2.putText(imgRGB, f'Score: {int(detection.score[0]*100)}',(20,70), cv2.FONT_HERSHEY_PLAIN, 2,(0, 255, 0), 2)
    if val == 1:
        cv2.putText(imgRGB, 'Status: Match_found', (20,150), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0), 2)
        val = 0
    cv2.imshow("Image",imgRGB)
    cv2.waitKey(1)
    #debug</>
