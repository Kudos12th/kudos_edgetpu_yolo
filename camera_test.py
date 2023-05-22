import cv2

def gstreamer_pipeline(
    capture_width=224,
    capture_height=224,
    display_width=224, #1280,
    display_height=224, #720,
    framerate=60,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

def show_camera():
    # To flip the image, modify the flip_method parameter (0 and 2 are the most common)
    img_count = 0 

    print(gstreamer_pipeline(flip_method=0))
    cap = cv2.VideoCapture(0)

    if cap.isOpened():

        while True: 
            ret_val, img = cap.read()
           # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            cv2.imshow("CSI Camera", img)

            keyCode = cv2.waitKey(1000) & 0xFF

            # Stop the program on the ESC key
            if keyCode == 27:
                break

            #if keyCode == ord('s'):
            	#img_count += 1
            	#save_file_name = './img/' + format(img_count, '04') + '.jpg'
            	#print(format(img_count, '04'))
            	#cv2.imwrite(save_file_name, img)

            img_count += 1
            save_file_name = './img/' + format(img_count, '04') + '.jpg'
            print(format(img_count, '04'))
            cv2.imwrite(save_file_name, img)

        cap.release()
        cv2.destroyAllWindows()
    else:
        print("Unable to open camera")


if __name__ == "__main__":
    show_camera()
