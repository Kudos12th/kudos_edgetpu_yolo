import cv2

def save_camera_images():
    cap = cv2.VideoCapture(0)

    # 카메라 해상도 설정
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if cap.isOpened():
        img_count = 1
        frame_rate = 1  # 이미지 저장 간격 (1초마다 저장)
        frame_interval = int(cap.get(cv2.CAP_PROP_FPS)) * frame_rate

        while True:
            ret, frame = cap.read()

            # cv2.imshow("Camera", frame)

            if img_count % frame_interval == 0:
                save_file_name = f"./chess/img__{img_count}.jpg"
                cv2.imwrite(save_file_name, frame)
                print(f"Saved {save_file_name}")

            img_count += 1

            # # 1초 대기
            # if cv2.waitKey(1) == ord('q'):
            #     break

        cap.release()
        cv2.destroyAllWindows()
    else:
        print("Unable to open camera.")


if __name__ == "__main__":
    save_camera_images()
