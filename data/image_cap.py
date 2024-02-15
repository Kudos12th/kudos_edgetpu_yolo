import cv2
import os
import time

def capture_and_save_images(camera_index, save_directory, images_per_second, total_images):
    # Open the camera
    cap = cv2.VideoCapture(camera_index)

    # Check if the camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Create the save directory if it doesn't exist
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    image_count = 0
    start_time = time.time()

    while image_count < total_images:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Save the image
        image_name = os.path.join(save_directory, f"{image_count}.jpg")
        cv2.imwrite(image_name, frame)

        # Display image count
        print(f"Saved image {image_count}")

        # Increment image count
        image_count += 1

        # Wait for the next frame
        time.sleep(1 / images_per_second)

    # Release the camera and close the window
    cap.release()
    cv2.destroyAllWindows()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Captured {total_images} images in {elapsed_time:.2f} seconds.")

# 설정
camera_index = 0  # 카메라 인덱스 (일반적으로 0 또는 1)
save_directory = "./data"  # 이미지 저장 디렉토리
images_per_second = 5  # 초당 저장할 이미지 수
total_images = 3000  # 총 저장할 이미지 수

# 이미지 캡처 및 저장 실행
capture_and_save_images(camera_index, save_directory, images_per_second, total_images)
