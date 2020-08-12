import shutil
from pathlib import Path

import cv2
import numpy as np

from swiss_army_tensorboard import tfboard_loggers

BASE_LOG_FOLDER_PATH = Path("test_logs")

if BASE_LOG_FOLDER_PATH.is_dir():
    shutil.rmtree(BASE_LOG_FOLDER_PATH, ignore_errors=False)
BASE_LOG_FOLDER_PATH.mkdir(parents=True)

############ Logging Text (Markdown) ##############

text_log_folder_path = BASE_LOG_FOLDER_PATH / "text_test"
text_logger = tfboard_loggers.TFBoardTextLogger(text_log_folder_path)

test_text_in_markdown = """
# Test

- This
- is
- a
- list

## Sub Heading

You can use **bold** text and also *italic*

### Sub-sub Heading

Wow, you can use [LINKS](https://gaborvecsei.com), and everything which is supported in **Markdown**
"""

text_logger.log_markdown("text_tag", test_text_in_markdown, -1)

############ Logging Text Continuously ##############

# Method 1:

continuous_text_log_folder_path = BASE_LOG_FOLDER_PATH / "continuous_m1_text_test"
continuous_text_logger = tfboard_loggers.TFBoardContinuousTextLogger(continuous_text_log_folder_path,
                                                                     "continuous_m1_text_tag")
continuous_text_logger.markdown("# Method 1")
continuous_text_logger.info("This is an info")
continuous_text_logger.error("This is an error")
continuous_text_logger.warn("This is a warning")

continuous_text_logger.stop_logging()

# Method 2:

continuous_text_log_folder_path = BASE_LOG_FOLDER_PATH / "continuous_m2_text_test"
with tfboard_loggers.TFBoardContinuousTextLogger(continuous_text_log_folder_path,
                                                 "continuous_m2_text_tag") as continuous_text_logger_m2:
    continuous_text_logger_m2.markdown("# Method 2")
    continuous_text_logger_m2.info("This is an info too")
    continuous_text_logger_m2.error("This is an error :(")
    continuous_text_logger_m2.warn("This is a warning, so pay attention")

############ Logging Scalars ##############

scalar_log_folder_path = BASE_LOG_FOLDER_PATH / "scalar_test"
scalar_logger = tfboard_loggers.TFBoardScalarLogger(scalar_log_folder_path)

for i, t in enumerate(np.arange(0.0, 1.0, 0.01)):
    val = np.sin(2 * np.pi * t)
    scalar_logger.log_scalar("scalar_tag", val, i)

############ Logging Histograms ##############

hist_log_folder_path = BASE_LOG_FOLDER_PATH / "histogram_test"
hist_logger = tfboard_loggers.TFBoardHistogramLogger(hist_log_folder_path)

for i in range(1000):
    val = np.random.rand(50) * (i + 1)
    hist_logger.log_histogram("hist_tag", val, i, bins=100)

############ Logging Images ##############

image_log_folder_path = BASE_LOG_FOLDER_PATH / "image_test"
image_logger = tfboard_loggers.TFBoardImageLogger(image_log_folder_path)

base_image = np.zeros((100, 100, 3), dtype=np.uint8)
for i in range(10):
    image_1 = cv2.circle(base_image.copy(), (50, 50), i * 3, (255, 0, 0))
    image_2 = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    image_logger.log_images("image_tag", [image_1, image_2], i)

##########################################

print("""
Now you can open Tensorboard to inspect the results:

$ tensorboard --logdir test_logs --port 6006
""")
