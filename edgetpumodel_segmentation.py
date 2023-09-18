"""An example using `BasicEngine` to perform semantic segmentation.

The following command runs this script and saves a new image showing the
segmented pixels at the location specified by `output`:

python3 examples/semantic_segmentation.py \
--model models/deeplabv3_mnv2_pascal_quant_edgetpu.tflite \
--input models/bird.bmp \
--keep_aspect_ratio \
--output ${HOME}/segmentation_result.jpg
"""


from edgetpu.basic.basic_engine import BasicEngine
from PIL import Image
import numpy as np
import numpy as np
from PIL import Image
from utils import ColorMap


class EdgeTPUSegmentation:

    def __init__(self, model_file):
        self.model_file = model_file

        # Initialize Edge TPU engine for YOLOv8
        self.engine = BasicEngine(model_file)
        _, self.model_height, self.model_width, _ = self.engine.get_input_tensor_shape()


    def segment_image(self, input_image_path):
        # Open and resize the input image
        input_image = Image.open(input_image_path)
        input_image = input_image.resize((self.model_width, self.model_height))

        # Convert the input image to a NumPy array
        input_tensor = np.asarray(input_image).flatten()

        # Run inference on the Edge TPU with YOLOv8
        _, raw_result = self.engine.run_inference(input_tensor)


        # Process the raw result using YOLOv8-specific logic
        result = np.reshape(raw_result, (self.model_height, self.model_width))
        segmented_image = self.process_raw_result(result)

        # Return the processed segmentation image
        return segmented_image
    

    def process_raw_result(self, raw_result):


        # Convert the segmentation result to a colored image using label_to_color_image
        vis_result = ColorMap.label_to_color_image(raw_result.astype(int)).astype(np.uint8)

        # Save the result image
        vis_image = Image.fromarray(vis_result)

        return vis_image



