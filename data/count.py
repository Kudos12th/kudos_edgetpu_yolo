import os
from PIL import Image

def rename_images(input_folder):
    # List all files in the input directory
    files = os.listdir(input_folder)

    for i, file in enumerate(files):
        # Check if the file is an image (you may want to add more image file extensions)
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            # Construct the full path to the input image
            input_path = os.path.join(input_folder, file)

            # Get the file extension
            file_extension = os.path.splitext(file)[1]

            # Construct the new filename (starting from 0)
            new_filename = f"{i}{file_extension}"

            # Construct the full path to the output image
            output_path = os.path.join(input_folder, new_filename)

            # Rename the image file
            os.rename(input_path, output_path)

            # Display a message
            print(f"Renamed {file} to {new_filename}")

# 설정
input_folder = "filtered_images"  # 이미지 폴더

# 이미지 파일 이름 재설정 실행
rename_images(input_folder)
