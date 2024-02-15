import os
from PIL import Image

def filter_and_save_images(input_folder, output_folder):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # List all files in the input directory
    files = os.listdir(input_folder)

    for file in files:
        # Check if the file is an image (you may want to add more image file extensions)
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            # Construct the full path to the input image
            input_path = os.path.join(input_folder, file)

            # Open the image using Pillow (PIL)
            image = Image.open(input_path)
            image_number = int(file.split('.')[0])

            # Check if the image number is a multiple of 5
            if image_number % 5 == 0:
                # Construct the full path to the output image
                output_path = os.path.join(output_folder, file)

                # Save the image to the output directory
                image.save(output_path)

                # Display a message
                print(f"Saved {file}")

# 설정
input_folder = "images"  # 입력 이미지 폴더
output_folder = "filtered_images"  # 10의 배수 이미지 저장 폴더

# 이미지 필터링 및 저장 실행
filter_and_save_images(input_folder, output_folder)
