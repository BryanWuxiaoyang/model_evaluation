# 持久化数据存储
## 介绍
该工具用于在处理大批量数据时，进行数据的持久化读取和存储（即将数据在磁盘和内存间交互），以及用于断点保存，防止因为不可预知原因造成的程序错误，以及随之的数据丢失。

## 使用例子
### 集合数据
目前实现了list和dict的自动保存，分别是```AutoSaveList```和```AutoSaveDict```,其使用方式与```list```和```dict```相同，且可以通过```save```和```load```在磁盘和内存间进行数据读取和保存。

```python
from AutoSave import AutoSaveList, AutoSaveDict

data1 = AutoSaveList(file_path="example_auto_save_list.json")
data1.append(1)
data1.append(2)
print(data1[0])
data1.save()
data1.load()
data1.clear()

data2 = AutoSaveDict(file_path="example_auto_save_dict.json")
data2.put("key", "value")
print(data2["key"])
data2.save()
data2.load()
data2.clear()
```

### 图像数据
#### ResultManager
该类本质上是```AutoSaveDict```，但其值的类型固定为```dict```，可以调整传入的参数。

```python
param1 = 'param1'
param2 = 'param2'

from ResultManager import ResultManager
manager = ResultManager(key_name='key_name', file_path="example_index_file.json")

manager.append(key="key1", param1=param1, param2=param2)

manager.save()
```

#### ImageResultManager
该类在保存```ResultManager```的数据的同时，可以同步保存对应的图像数据到图像文件夹中：
```python
param1 = 'param1'
param2 = 'param2'

from ResultManager import ImageResultManager
import cv2

manager = ImageResultManager(index_file_path="example_index_file.json", image_dir="example_images/")
image = cv2.imread("test.jpg", cv2.IMREAD_COLOR)

manager.append(image=image, image_name="test.jpg", param1=param1, param2=param2)

manager.save()
```

#### 迭代器
```AutoSaveIterator```用于包装原始迭代器，使得数据在被读取的过程中能够自动进行存储和读取，从而实现断点续传的功能（该功能在程序需要长时间运行处理大批量数据时，可以避免因为未知错误造成的内存数据丢失）。
```AutoSaveIterable```用于包装原始```Iterable```，每次读取时会通过```__iter__```方法生成一个```AutoSaveIterator```迭代器。

```python
from AutoSave import AutoSaveList, AutoSaveIterator

input_data = [i for i in range(10000)]
output_data = AutoSaveList(file_path="example_auto_save_list.json")

input_iterator = AutoSaveIterator(
    iterator_generator=lambda iter_cnt: input_data[iter_cnt:].__iter__(),
    resume_iter_file_path="example_resume_iter",
    loader=None,
    saver=lambda iter_cnt: output_data.save(),
    save_every=1000,
)

for data in input_iterator:
    output_data.append(data)

input_iterator.save()
```