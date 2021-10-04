import numpy as np
import cv2
import os


def list_file_paths(dir_path: str):
    for sub_dir_name in os.listdir(dir_path):
        sub_dir_path = os.path.join(dir_path, sub_dir_name)
        if os.path.isdir(sub_dir_path):
            for file_path in list_file_paths(sub_dir_path):
                yield file_path
        else:
            yield sub_dir_path


def make_sub_image_name(image_name: str, bbox: list, postfix: str = ".jpg"):
    points_str = ""
    for point in bbox:
        points_str += str(int(point))
    image_name = image_name + "_" + points_str + postfix
    return image_name


def make_boxed_image(image: np.ndarray, boxes: list):
    in_boxes = []
    # print("boxes: ", boxes)
    for box in boxes:
        in_boxes.append(np.asarray([[box[0], box[1]],
                                    [box[2], box[3]],
                                    [box[4], box[5]],
                                    [box[6], box[7]]
                                    ], dtype=np.int32))
    new_image = image.copy()
    return cv2.drawContours(new_image, in_boxes, -1, (0, 255, 255), 2)


def flatten_box(box: list):
    return [box[0][0], box[0][1], box[1][0], box[1][1], box[2][0], box[2][1], box[3][0], box[3][1]]


def wrap_box(box: list):
    return [[box[0], box[1]], [box[2], box[3]], [box[4], box[5]], [box[6], box[7]]]


def get_recognition_result(results):
    """
    generate (text, confidence) as recognition result
    :param results: [(text, confidence)]
    :return: (text, confidence)
    """

    max_confidence = 0
    max_text = None

    for text, confidence in results:
        if confidence >= max_confidence:
            max_confidence = confidence
            max_text = text

    return max_text, max_confidence


def create_dir(dir_path: str):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    return dir_path


def create_and_clear_dir(dir_path: str):
    import shutil
    if os.path.exists:
        shutil.rmtree(dir_path)
    os.mkdir(dir_path)
    return dir_path


def load_value(file_path: str, default=None):
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf8") as f:
            ret = f.readline()
    else:
        ret = 0

    return ret


def save_value(file_path: str, value):
    with open(file_path, "w", encoding="utf8") as f:
        f.write(value)


def count_file_num(path: str):
    if not os.path.exists(path):
        return 0
    elif os.path.isfile(path):
        return 1
    else:
        cnt = 0
        for sub_name in os.listdir(path):
            sub_path = os.path.join(path, sub_name)
            cnt += count_file_num(sub_path)
        return cnt


def load_example_image_infos():
    root = os.path.dirname(__file__)
    file_path = os.path.join(root, "ocr_410_anno/gt/Label.txt")
    image_infos = []
    with open(file_path, "r", encoding='utf8') as f:
        for line in f.readlines():
            image_name = line.split()[0]
            # print("loading image: ", image_name)
            import json
            attrs = json.loads(line[len(image_name):])

            image_name = image_name.split('/')[1]
            image_path = os.path.join(os.path.join(root, "ocr_410_anno/testsets"), image_name)
            boxes = []
            for attr in attrs:
                text = attr['transcription']
                shape = attr['points']
                shape = [shape[0][0], shape[0][1], shape[1][0], shape[1][1], shape[2][0], shape[2][1], shape[3][0],
                         shape[3][1]]
                confidence = 1.0
                boxes.append((text, shape, confidence))
            image_infos.append((image_path, boxes))
            # print("image info: ", (image_path, boxes))
    return image_infos


def create_image_infos(image_dir: str, model, image_info_path: str = None,
                       worker_num=0, group_size=1024,
                       resume_iter_file_path: str = None, save_every=10000, restart: bool = False):
    """
    Creating [image_infos] dataset by applying each image in [image_dir] to model, and storing results into [image_infos].

    :param restart:
    :param image_dir:
    :param model:
    :param image_info_path:
    :param worker_num: Number of parallel threads. Serial if set to 0 or 1.
    :param group_size: Number of iterations before collecting results each time.
    :param resume_iter_file_path: Logger file recording iteration counts, used for breakpoint-resuming.
    :param save_every:
    :return: [image_infos] generated from model
    """
    import concurrent.futures as concur
    from AutoSave import AutoSaveList, AutoSaveIterator
    from SimpleExecutor import SimpleExecutor
    executor = concur.ThreadPoolExecutor(worker_num) if worker_num >= 2 else SimpleExecutor()
    futures = []
    file_paths = [file_path for file_path in list_file_paths(image_dir)]

    if resume_iter_file_path is not None:
        auto_save = True
        image_infos = AutoSaveList(image_info_path, lazy_load=True, print_info=True)
        images = AutoSaveIterator(
            iterator_generator=lambda iter_cnt: file_paths[iter_cnt:].__iter__(),
            resume_iter_file_path=resume_iter_file_path,
            loader=None,
            saver=lambda iter_cnt: image_infos.save(),
            save_every=save_every,
            auto_iter=False,
            auto_save=True,
            restart=restart,
            print_info=True
        )
        print("resume_iter: ", images.current_iter)
    else:
        auto_save = False
        image_infos = []
        images = file_paths

    group_cnt = 0
    total_cnt = 0
    processed_cnt = [0]

    def do(img_path):
        processed_cnt[0] += 1
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image is None:
            print("Warning: image is none: ", img_path)
            return None, None
        return img_path, model(image)

    for image_path in images:
        print(total_cnt, image_path)
        # print("generating image: ", image_path)
        futures.append(executor.submit(lambda img_path: do(img_path), image_path))
        group_cnt += 1
        total_cnt += 1
        if group_cnt >= group_size:
            for future in futures:
                try:
                    image_path, detection_result = future.result()
                except TimeoutError:
                    print("timeout: ", image_path)
                    image_path, detection_result = None, None
                if image_path is not None:
                    image_infos.append((image_path, detection_result))
                if auto_save:
                    images.try_iter()

            futures.clear()
            group_cnt = 0

    for future in futures:
        try:
            image_path, detection_result = future.result()
        except TimeoutError:
            print("timeout: ", image_path)
            image_path, detection_result = None, None
        if image_path is not None:
            image_infos.append((image_path, detection_result))
        if auto_save:
            images.try_iter()

    futures.clear()
    group_cnt = 0

    if auto_save:
        images.save()
        image_infos.load()
        return image_infos.data
    else:
        return image_infos


def cv2ImgAddText(img, text, x, y, textColor=(0, 255, 0), textSize=20):
    from PIL import Image, ImageDraw, ImageFont
    cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype("font/simsun.ttc", textSize, encoding="utf-8")
    # 绘制文本
    draw.text((x, y), text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def make_texted_image(image: np.ndarray, texts: list) -> np.ndarray:
    """
    Attack [texts] at the bottom of the original [image]
    """
    font_height = 20
    font_width = 500

    new_height = image.shape[0] + len(texts) * font_height
    new_width = max(image.shape[1], font_width)
    new_image = np.zeros((new_height, new_width, image.shape[2]), dtype=np.uint8)
    for h in range(image.shape[0]):
        for w in range(image.shape[1]):
            for c in range(image.shape[2]):
                new_image[h][w][c] = image[h][w][c]

    for (idx, text) in enumerate(texts):
        x = 0
        y = image.shape[0] + idx * font_height
        new_image = cv2ImgAddText(new_image, text, x, y)

    return new_image


def make_sub_image(image: np.ndarray, bbox: list) -> np.ndarray:
    min_x = int(min(bbox[0], bbox[2], bbox[4], bbox[6]))
    min_y = int(min(bbox[1], bbox[3], bbox[5], bbox[7]))
    max_x = int(max(bbox[0], bbox[2], bbox[4], bbox[6]))
    max_y = int(max(bbox[1], bbox[3], bbox[5], bbox[7]))

    sub_image = image[min_y: max_y, min_x: max_x, :]
    if max_x - min_x > 0 and max_y - min_y > 0:
        assert sub_image.shape[0] == max_y - min_y
        assert sub_image.shape[1] == max_x - min_x
    return sub_image


def load_video(video_path: str, time_interval: float = 1.0, start_ratio: float = 0.0, end_ratio: float = 1.0):
    """
    Extracting frames from video[start_ratio: end_ratio].

    :return: Iterator containing extracted frames.
    """
    import cv2
    import math
    capture = cv2.VideoCapture(video_path)
    frame_num = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_start = int(frame_num * start_ratio)
    frame_end = int(frame_num * end_ratio)
    print("frame_num: ", frame_num)

    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_interval = int(math.ceil(fps * time_interval))
    frame_offset = frame_start
    while True:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_offset)
        suc, frame = capture.read()
        if suc is False:
            break
        # print("yield frame: ", frame_offset)
        yield frame
        frame_offset += frame_interval
        if frame_offset > frame_end:
            break
