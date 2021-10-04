# OCR模型处理工具
## 介绍
此项目主要可用于以下用途：
- 对OCR检测模型和识别模型进行测试、评价。
- 利用OCR模型生成数据集标签
- 对数据集进行处理（数据清洗、筛选、生成等）

程序进行了并行优化，使得模型可以在多线程模式下高效地处理。

## 使用示例
### 安装依赖

```
pip install -r requirements.txt
```

### 测试模型
首先读取测试集的数据：
```python
import utils
image_infos = utils.load_example_image_infos()
```
然后初始化需要进行测试的模型(这里以Gicp模型为例，该模型可分别进行检测和识别）：
```python
from models import GicpRecognizer
model = GicpRecognizer()
```
**注：该模型作为服务，通过rpc调用，因此需要在服务器中运行**

之后调用对应的测试函数，得到测试结果：
```python
# Evaluate detection
from model_evaluation import evaluate_detection_model
detection_result = evaluate_detection_model(model_name='gicp', model=model, image_infos=image_infos)

# Evaluate recognition
from model_evaluation import evaluate_recognition_model
recognition_result = evaluate_recognition_model(model_name='gicp', model=model, image_infos=image_infos)

# Evaluate both
from model_evaluation import evaluate_detection_and_recognition_model 
detection_result, recognition_result = evaluate_detection_and_recognition_model(model_name='gicp', model=model, image_infos=image_infos)
```

### 生成数据集标签
在拥有原始数据后，可以通过指定的模型对数据进行推理并生成数据集标签：
```python
from utils import create_image_infos
from models import GicpRecognizer

image_dir = "ocr_410_anno/testsets"
model = GicpRecognizer()
image_infos_path = "test_image_infos.json"
image_infos = create_image_infos(image_dir=image_dir, model=model, image_infos_path=image_infos_path)
```
得到的数据集标签可以用于后续对模型的测试和数据生成。

### 数据处理
```evaluate_model```函数除了可以实现对模型的测试外，还可以对```image_infos```数据进行读取和处理。
实际上，其会分别对**主图像**回调自定义的检测处理函数，以及对通过```image_infos```得到的**识别子图**回调自定义的识别处理函数。
二者的入参都为```args: Munch```，具体的参数情况可参考```evaluate_model```的注释。

例如若想要利用数据集标签，将标签中的识别子图提取出来，则可以如下处理：
```python
from utils import load_example_image_infos
from model_evaluation import evaluate_model
from models import NoRecognizer # 该模型为空模型，表示不对数据进行推理，返回的推理结果均为空

image_infos = load_example_image_infos()
model = NoRecognizer()
sub_images = []

# 定义识别回调函数
def recognition_preprocessor(args):
    print("submit: ", args.image_path)

def recognition_postprocessor(args):
    print("process: ", args.image_path)
    sub_images.append(args.sub_image)
    
# 调用测试函数
evaluate_model(model_name='none', model=model, image_infos=image_infos)

print(len(sub_images))
```

若想要把识别子图及其识别结果一同进行存储，可以利用```ResultManager.ImageResultManager```来进行管理：

```python
from utils import load_example_image_infos
from model_evaluation import evaluate_model
from models import GicpRecognizer
from ResultManager import ImageResultManager
from utils import create_and_clear_dir

image_infos = load_example_image_infos()
model = GicpRecognizer()

manager = ImageResultManager(index_file_path="example_labels.json", image_dir=create_and_clear_dir("example_image_dir"), lazy_load=False)

# 定义识别回调函数
def recognition_preprocessor(args):
    print("submit: ", args.image_path)

def recognition_postprocessor(args):
    print("process: ", args.image_path)
    manager.append(args.sub_image, args.image_path, text=args.text, pattern=args.pattern, ratio=args.ratio)

# 调用测试函数
evaluate_model(model_name='gicp', model=model, image_infos=image_infos,
recognition_preprocessor=recognition_preprocessor,
recognition_postprocessor=recognition_postprocessor)

# 保存数据
manager.save(clear=True)
```

### 其它应用
将检测结果和识别结果附加到原始图像上，并保存：
```python
from utils import *
from model_evaluation import evaluate_model
from models import GicpRecognizer
from ResultManager import ImageResultManager
from utils import create_and_clear_dir
import cv2
import os

detection_image_dir = create_and_clear_dir("example_detection_image_dir")
recognition_image_dir = create_and_clear_dir("example_recognition_image_dir")

image_infos = load_example_image_infos()
model = GicpRecognizer()

def detection_postprocessor(args):
    boxes = [box for _, box, _ in args.result]
    image = make_boxed_image(args.image, boxes)
    image_name = os.path.basename(args.image_path)
    cv2.imwrite(os.path.join(detection_image_dir, image_name), image)

def recognition_postprocessor(args):
    image = make_texted_image(args.sub_image, ["text: "+args.text, "pattern: ", args.pattern])
    image_name = make_sub_image_name(os.path.basename(args.image_name), args.bbox)
    cv2.imwrite(os.path.join(recognition_image_dir, image_name), image)

evaluate_model("", model, image_infos, detection_postprocessor=detection_postprocessor, recognition_postprocessor=recognition_postprocessor)
    
```