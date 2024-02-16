import cv2
import os
from datetime import datetime
import numpy as np

now = datetime.now()
formatted_now = now.strftime('%m-%d_%H-%M')
save_path = f'./captured_img_dataset/{formatted_now}'
os.makedirs(save_path, exist_ok=True)

cap = cv2.VideoCapture(0) 

n = 1
last_time = datetime.now()

def camera_calibration(frame):
    cameraMatrix = np.array([[9.4088597780774421e+02, 0., 3.7770158111216648e+02],
                            [0., 9.4925081349933703e+02, 3.2918621409818121e+02],
                            [0., 0., 1.]])
    distCoeffs = np.array([-4.4977607383629226e-01, -3.0529616557684319e-01,
                        -3.9021603448837856e-03, -2.8130335366792153e-03,
                        1.2224960045867554e+00])
    cal_frame = cv2.undistort(frame, cameraMatrix, distCoeffs) #, None, new_img_size)
    return cal_frame


while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    frame = camera_calibration(frame)  # 카메라 왜곡 보정
    cv2.imshow('Camera', frame)

    if (datetime.now() - last_time).total_seconds() >= 0.5:
        img_name = f"{save_path}/{n}.jpg"
        cv2.imwrite(img_name, frame)
        print(f"Saved: {img_name}")
        n += 1
        last_time = datetime.now()

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()