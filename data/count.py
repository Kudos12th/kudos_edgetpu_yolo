import os


def rename_images(input_folder, output_folder, start_number):
    
    # List all files in the input directory
    files = os.listdir(input_folder)
    files = sorted(files)
    print(files)

    for file in files:
        # Check if the file is an image or text file
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.txt')):
            # Get the file extension
            file_extension = os.path.splitext(file)[1]

            # Convert the current file name (excluding extension) to an integer
            try:
                current_number = int(os.path.splitext(file)[0])
            except ValueError:
                print(current_number)
                # Skip if the filename cannot be converted to an integer
                continue

            # Calculate the new number by adding start_number
            new_number = current_number + start_number

            # Construct the new filename
            new_filename = f"{new_number}{file_extension}"

            # Construct the full path to the input file
            input_path = os.path.join(input_folder, file)

            # Construct the full path to the output file
            output_path = os.path.join(output_folder, new_filename)

            # Rename the file
            os.rename(input_path, output_path)

            # Display a message
            print(f"Renamed {file} to {new_filename}")

# 설정
input_folder = "img5"  # 이미지 폴더
output_folder = "result"
start_number = 5211 - 699 # 시작 번호

# 이미지 및 텍스트 파일 이름 재설정 실행
rename_images(input_folder, output_folder, start_number)



