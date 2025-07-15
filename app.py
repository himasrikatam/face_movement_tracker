# step1: Install the required packages
import cv2
import mediapipe as mp
import numpy as np
import time
# mediapipe works only python 3.10 and below
# C:\Python310\python.exe -m venv venv
# step2: mediapipe facemesh setup
mp_face_mesh = mp.solutions.face_mesh # defining mediapipe face mesh module(468 landmarks)
mp_drawing = mp.solutions.drawing_utils #draws dots(face mesh)
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) #
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1) #thickness of the lines and radius of the dots
# step3: start webcam and get the face mesh results
cap = cv2.VideoCapture(0) #starts webcam
while cap.isOpened():
    success, image = cap.read()
    start = time.time()
    # Flip the image horizontally for a later selfie-view display
    # convert the BGR image to RGB as mediapipe uses RGB images but OpenCV uses BGR images
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    # why???????//
    image.flags.writeable = False
    # get the face mesh results
    results = face_mesh.process(image)
    # To improve performance
    image.flags.writeable = False
    # convert the RGB image to BGR so opencv can display it
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    img_h, img_w, img_c = image.shape
    face_3d = []
    face_2d = []
    # step4: get the face mesh landmarks
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
            #   1:nose tip, 33: left eye, 61: right eye, 199: left ear, 263: right ear, 291: chin
              if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                if idx == 1:
                   nose_2d = (int(lm.x * img_w), int(lm.y * img_h))
                   nose_3d = (int(lm.x * img_w), int(lm.y * img_h), lm.z* 3000)
                # convert landmarks to 2D and 3D coordinates   
                x, y = int(lm.x * img_w), int(lm.y * img_h)
                face_2d.append([x, y])
                face_3d.append([x, y, lm.z])
            # convert the face mesh landmarks to numpy array
            face_2d = np.array(face_2d, dtype=np.float32)
            face_3d = np.array(face_3d, dtype=np.float32)
            # solve PnP=> estimate the pose of the face
            focal_length = 1 * img_w
            cam_matrix= np.array([[focal_length, 0, img_w / 2],
                                       [0, focal_length, img_h / 2],
                                       [0, 0, 1]])
            dist_matrix = np.zeros((4, 1), dtype=np.float64)
            success, rotation_vector, translation_vector = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)    
            # get the rotation matrix
            rmat, jac = cv2.Rodrigues(rotation_vector)
            # get angles
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
            # get y rotation degree
            x = angles[0] * 360
            y = angles[1] * 360 
            z = angles[2] * 360
            # see which direction the face is looking
            if y < -10:
                text = "Looking left"
            elif y > 10:
                text = "Looking right"
            else:
                text = "Forward"
            # display the nose direction
            nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rotation_vector, translation_vector, cam_matrix, dist_matrix)
            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_2d[0]+y*10), int(nose_2d[1]-x*10))
            cv2.line(image, p1, p2, (255, 0, 0), 3)
            # add the text to the image on the top left corner
            cv2.putText(image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
            cv2.putText(image,"x: "+ str(np.round(x, 2)), (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(image,"y: "+ str(np.round(y, 2)), (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(image,"z: "+ str(np.round(z, 2)), (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        end = time.time()
        total_time = end - start
        fps = 1 / total_time
        cv2.putText(image, f'FPS: {int(fps)}', (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)   
        # draw the face mesh landmarks
        mp_drawing.draw_landmarks(
           image=image,
            landmark_list=face_landmarks,
            connections = mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=drawing_spec,
            connection_drawing_spec=drawing_spec
        )
    cv2.imshow('Face Movement Tracker', image)
    if cv2.waitKey(5) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()