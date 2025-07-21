# step1: Install the required packages
import cv2
import mediapipe as mp
import numpy as np
import time
# mediapipe works only python 3.10 and below
# C:\Python310\python.exe -m venv venv
# step2: mediapipe facemesh setup
mp_face_mesh = mp.solutions.face_mesh # defining mediapipe face mesh module(468 landmarks)
mp_drawing = mp.solutions.drawing_utils # draws dots(face mesh)
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) # The model will register a face if it's at least 50% confident
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1) #thickness of the lines and radius of the dots
# step3: start webcam and get the face mesh results
cap = cv2.VideoCapture(0) #starts webcam 0-defualt camera, 1-external camera
while cap.isOpened():
    success, image = cap.read()
    start = time.time()
    # Preprocess Image for MediaPipe
    # Flip the image horizontally for a later selfie-view display
    # convert the BGR image to RGB as mediapipe uses RGB images but OpenCV uses BGR images
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB) # 0: vertical flip, 1: horizontal flip, -1: both flips
    #first reads the images only, after detecting the landmarks it allows writing on the image(True)
    image.flags.writeable = False
    # get the face mesh results (Detects face landmarks)
    results = face_mesh.process(image)
    # To improve performance (allows writing on the image(True))
    image.flags.writeable = True  
    # convert the RGB image to BGR so opencv can display it
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    img_h, img_w, img_c = image.shape # Gets image dimensions
    face_3d = []
    face_2d = []
    # step 4: get the face mesh landmarks
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark): #  pairs each landmark with its index (0 to 467)
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
            # Camera intrinsic matrix used - Converts 3D world points → 2D image pixels.
            cam_matrix= np.array([[focal_length, 0, img_w / 2],
                                       [0, focal_length, img_h / 2],
                                       [0, 0, 1]])
            dist_matrix = np.zeros((4, 1), dtype=np.float64) # distortion coefficients: lens distortions(0 = ideal camera lens)
            # solvePnP: estimate the 3D pose (rotation + translation) of an object from its 2D image points
            success, rotation_vector, translation_vector = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)     # OUTPUT: 3x1 representation of rotation
            # get the rotation matrix - Calculate Rotation Angles
            rmat, jac = cv2.Rodrigues(rotation_vector) # converted to a 3x3 rotation matrix 
            # get angles
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
            # get y rotation degree - In MediaPipe/OpenCV’s right-handed system: X = right (→), Y = down (↓), Z = outward (📷→)
            x = angles[0] * 360 # pitch(up/down)
            y = angles[1] * 360 # yaw(left/right)
            z = angles[2] * 360 # roll(tilt)
            # see which direction the face is looking
            # Check YAW (left/right)
            if y < -10:
                text = "Looking left"
            elif y > 10:
                text = "Looking right"
            else:
                text = "Forward"
            # Check PITCH (up/down)
            if x < -15:
              text = "Looking Down"
            elif x > 15:
               text = "Looking Up"
            # Check ROLL (tilt)
            if z < -10:
              text = "Head Tilted Right"
            elif z > 10:
             text = "Head Tilted Left"
            # display the nose direction
            nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rotation_vector, translation_vector, cam_matrix, dist_matrix)
            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_2d[0]+y*10), int(nose_2d[1]-x*10))
            cv2.line(image, p1, p2, (255, 0, 0), 3)
            # add the text to the image on the top left corner
        #     cv2.putText(
        #     image,       # Image to draw on
        #     text,        # Text string to display
        #     (x, y),      # Bottom-left corner of text (20,50) 20 px from left, 50 px from to
        #     font,        # Font type (e.g., cv2.FONT_HERSHEY_SIMPLEX)
        #     fontScale,   # Font size multiplier eg. 2 => 2x default size
        #     (B, G, R),   # Text color (Blue, Green, Red)
        #     thickness,   # Thickness of text strokes
        #     lineType     # Optional: Type of line (default=8)
        #     )
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
    if cv2.waitKey(5) & 0xFF == 27: # Checks for keyboard input every 5 milliseconds, 27=ASCII code for the ESC key. 0xFF is hexadecimal for 255 & 0xFF is used to mask the last 8 bits of the integer.
        break
cap.release()
cv2.destroyAllWindows()

    #       ┌───────────────────────────────────────┐
    #       │        Start Webcam (0)               │
    #       └───────────────────────┬───────────────┘
    #                               │
    #                               ▼
    #       ┌───────────────────────────────────────┐
    #       │    1. Capture Frame                   │
    #       │    2. Flip Horizontally (Mirror)      │
    #       │    3. Convert BGR → RGB               │
    #       └───────────────────────┬───────────────┘
    #                               │
    #                               ▼
    #       ┌───────────────────────────────────────┐
    #       │    Detect Face Mesh?                  │
    #       └───────────────────────┬───────────────┘
    #                    ┌──────────┴──────────┐
    #                    ▼                     ▼
    #   ┌───────────────────────┐    ┌───────────────────────┐
    #   │ No Face Found         │    │ Face Found            │
    #   │ → Skip Frame          │    │ → Get Key Landmarks   │
    #   └───────────────────────┘    │ (Nose, Eyes, Chin)    │
    #                               └──────────┬──────────┘
    #                                          │
    #                                          ▼
    #       ┌───────────────────────────────────────┐
    #       │    1. Convert to 2D/3D Points        │
    #       │    2. Calculate Head Pose (solvePnP) │
    #       │    3. Get Rotation Angles (Yaw/Pitch/Roll) │
    #       └───────────────────────┬───────────────┘
    #                               │
    #                               ▼
    #       ┌───────────────────────────────────────┐
    #       │ Check Direction:                      │
    #       │ - Yaw (Left/Right)                   │
    #       │ - Pitch (Up/Down)                    │
    #       │ - Roll (Tilt)                        │
    #       └───────────────────────┬───────────────┘
    #                               │
    #                               ▼
    #       ┌───────────────────────────────────────┐
    #       │ Display:                              │
    #       │ 1. Direction Text                     │
    #       │ 2. FPS Counter                        │
    #       │ 3. Face Mesh (Optional)               │
    #       └───────────────────────┬───────────────┘
    #                               │
    #                               ▼
    #       ┌───────────────────────────────────────┐
    #       │ Press ESC? → Yes: Exit                │
    #       │       ↓                               │
    #       │ No: Continue Loop                     │
    #       └───────────────────────────────────────┘