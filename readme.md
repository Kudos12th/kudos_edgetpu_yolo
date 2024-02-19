## Pycoral 🐚

https://coral.ai/docs/accelerator/get-started/#2-install-the-pycoral-library


```
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -

sudo apt-get update

sudo apt-get install libedgetpu1-std

sudo apt-get install python3-pycoral
```

<br>

## Setup 🔨

```
sudo apt-get update && sudo apt-get -y upgrade
sudo apt-get install -y git curl gnupg


## Get Python dependencies

sudo apt-get install -y python3 python3-pip
pip3 install --upgrade pip setuptools wheel


## ERROR: launchpadlib 1.10.13 requires testresources, which is not installed. 에러 발생 시 아래 코드 수행

// sudo apt install python3-testresources
python3 -m pip install numpy
python3 -m pip install opencv-python-headless
python3 -m pip install tqdm pyyaml


## Clone this repository

git clone https://github.com/Kudos12th/kudos_edgetpu_yolo.git
cd kudos_edgetpu_yolo
```

<br>

## Run the test script

1. roscore
2. change default camera number
    ```
    sudo apt install v4l-utils
    
    v4l2-ctl --list-devices
    
    parser.add_argument("--device", type=int, default=0, help="Image capture device to run live detection")
    ```
4. coral
5. detect
    ```
    python3 detect.py --model 3class_new-int8_edgetpu.tflite --stream
    python3 detect.py --model 3class_new_saved_model --stream
    ```


<br>

## Reference
- https://github.com/jveitchmichaelis/edgetpu-yolo

