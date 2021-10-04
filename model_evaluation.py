import os

import cv2
import numpy as np
from models import *
from utils import *
from SimpleExecutor import SimpleExecutor
from DetectionMetric import *
from RecognitionMetric import *
from munch import Munch
from ResultManager import *
from CallerChain import CallerChain
from BoxImageWriter import BoxImageWriter
import concurrent.futures as concur
from AutoSave import *


def _recognize_sub_image(sub_image: np.ndarray,
                         image: np.ndarray, image_path: str, bbox: list,
                         model, model_name: str,
                         text: str,
                         detection_index: int, recognition_index: int, total_index: int,
                         preprocessor=None, postprocessor=None):
    try:
        if preprocessor is not None: preprocessor(Munch(sub_image=sub_image,
                                                        image=image, image_path=image_path, bbox=bbox,
                                                        model=model, model_name=model_name,
                                                        text=text,
                                                        detection_index=detection_index,
                                                        recognition_index=recognition_index,
                                                        total_index=total_index))

        if postprocessor is not None:
            results = model.recognize(sub_image) if hasattr(model, "recognize") else model(sub_image)

            max_pattern = ""
            max_ratio = 0.0
            for pattern, points, confidence in results:
                from difflib import SequenceMatcher
                ratio = SequenceMatcher(None, pattern, text).ratio()
                if ratio >= max_ratio:
                    max_pattern = pattern
                    max_ratio = ratio

            return Munch(sub_image=sub_image,
                         image=image, image_path=image_path, bbox=bbox,
                         model=model, model_name=model_name,
                         result=results,
                         text=text, pattern=max_pattern, ratio=max_ratio,
                         detection_index=detection_index, recognition_index=recognition_index, total_index=total_index)
        else:
            return None
    except:
        print("recognition error occurred!")
        raise RuntimeError


def _detect_image(image: np.ndarray, image_path: str,
                  model, model_name: str,
                  gts: list,
                  detection_index: int, total_index: int,
                  preprocessor=None, postprocessor=None):
    try:
        if preprocessor is not None: preprocessor(Munch(image=image, image_path=image_path,
                                                        model=model, model_name=model_name,
                                                        gts=gts,
                                                        detection_index=detection_index, total_index=total_index))

        if postprocessor is not None:
            result = model.detect(image) if hasattr(model, "detect") else model(image)

            return Munch(image=image, image_path=image_path,
                         model=model, model_name=model_name,
                         gts=gts, result=result,
                         detection_index=detection_index, total_index=total_index)
        else:
            return None
    except:
        print("detection error occurred!")
        raise RuntimeError


def _receive_futures(futures: list, postprocessor=None):
    for future in futures:
        result = future.result()
        if result is not None and postprocessor is not None:
            postprocessor(result)
    futures.clear()


def evaluate_model(model_name, model, image_infos,
                   worker_num: int = 0, group_size: int = 1024000,
                   detection_preprocessor=None, detection_postprocessor=None,
                   recognition_preprocessor=None, recognition_postprocessor=None, ):
    """
    Run the OCR evaluation with given model with high efficiency, serving as the base framework for other appropriate
    evaluation actions such as detection evaluation(MAP).
    :param model_name: The alias name of the model
    :param model: model instance. It should have any one of [__call__], [detect] or [recognize] methods, takeing
    [image: numpy.ndarray] as input and producing [result: list<(text, bounding_box, confidence)>] as output.
    :param image_infos: The input images
    :param worker_num: Number of worker threads in parallel, serialized if set to 0
    :param group_size: Number of detection/recognition inference before collecting results each time
    :param detection_preprocessor: Actions before detection inference
    Action will be run by action(args) with args having attributes:
        image: image data
        image_path
        model
        model_name
        gts: list<(text, bounding_box, confidence)>, ground truth
        detection_index: Identify specific detection action
        total_index: Identify specific (detection+recognition) action

    :param detection_postprocessor: Actions after detection inference
    Action will be run by action(args) with args having attributes:
        image: image data
        image_path
        model
        model_name
        gts: list<(text, bounding_box, confidence)>, ground truth
        result: list<(text, bounding_box, confidence)>, detection result(as only [bounding_box] and [confidence] fields are useful)
        detection_index: Identify specific detection action
        total_index: Identify specific (detection+recognition) action

    :param recognition_preprocessor: Actions before recognition inference
    Action will be run by action(args) with args having attributes:
        image: image data
        image_path
        model
        model_name
        text: The ground truth text for recognition
        detection_index: Identify specific detection action
        recognition_index: Identify specific recognition action
        total_index: Identify specific (detection+recognition) action

    :param recognition_postprocessor: Actions after recognition inference
    Action will be run by action(args) with args having attributes:
        image: image data
        image_path
        model
        model_name
        result: list<(text, bounding_box, confidence)>, recognition result(as only [text] and [confidence] fields are useful)
        text: The ground truth text for recognition
        pattern: The text string recognized by [model]
        ratio: The edit-distance ratio between [pattern] and [text]. The higher the ratio, the closer [pattern] is to [text]
        detection_index: Identify specific detection action
        recognition_index: Identify specific recognition action
        total_index: Identify specific (detection+recognition) action
    """
    print("model_name: ", model_name, ", worker_num: ", worker_num,
          ", detection_enabled: ", str((detection_preprocessor is not None) or (detection_postprocessor is not None)),
          ", recognition_enabled: ",
          str((recognition_preprocessor is not None) or (recognition_postprocessor is not None))
          )

    from concurrent.futures import ThreadPoolExecutor
    executor = SimpleExecutor() if worker_num <= 1 else ThreadPoolExecutor(worker_num)

    detection_futures = []
    detection_index = 0

    recognition_futures = []

    total_index = 0

    group_cnt = 0

    for image_path, gt_boxes in image_infos:
        assert os.path.exists(image_path)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        if detection_preprocessor is not None or detection_postprocessor is not None:
            detection_futures.append(executor.submit(_detect_image,
                                                     image=image, image_path=image_path,
                                                     model=model, model_name=model_name,
                                                     gts=gt_boxes,
                                                     detection_index=detection_index, total_index=total_index,
                                                     preprocessor=detection_preprocessor, postprocessor=detection_postprocessor))
            group_cnt += 1

        detection_index += 1
        total_index += 1

        recognition_index = 0
        if recognition_preprocessor is not None or recognition_postprocessor is not None:
            for gt_box in gt_boxes:
                text = gt_box[0]
                points = gt_box[1]

                sub_image = make_sub_image(image, points)
                if sub_image.shape[0] > 0 and sub_image.shape[1] > 0:
                    recognition_futures.append(
                        executor.submit(_recognize_sub_image,
                                        sub_image=sub_image,
                                        image=image, image_path=image_path, bbox=points,
                                        model=model, model_name=model_name,
                                        text=text,
                                        detection_index=detection_index, recognition_index=recognition_index,
                                        total_index=total_index,
                                        preprocessor=recognition_preprocessor, postprocessor=recognition_postprocessor))
                    group_cnt += 1

                recognition_index += 1
                total_index += 1
        else:
            recognition_index += len(gt_boxes)
            total_index += len(gt_boxes)

        if group_cnt >= group_size:
            _receive_futures(detection_futures, detection_postprocessor)
            _receive_futures(recognition_futures, recognition_postprocessor)
            group_cnt = 0

    _receive_futures(detection_futures, detection_postprocessor)
    _receive_futures(recognition_futures, recognition_postprocessor)


def evaluate_detection_model(model_name, model, image_infos, worker_num: int = 0, group_size: int = 1024000,
                             print_info: bool = False):
    """
    Evaluating detection models using MAP algorithm.
    model can either implement [__call__] or [detect] method
    """
    metric = DetectionMetric()

    def detection_postprocessor(args):
        if print_info: print("detect ", args.detection_index, args.image_path)
        metric.append([bbox for (text, bbox, confidence) in args.gts],
                      [(bbox, confidence) for (text, bbox, confidence) in args.result])

    evaluate_model(model_name=model_name, model=model, image_infos=image_infos, worker_num=worker_num,
                   group_size=group_size,
                   detection_postprocessor=detection_postprocessor)

    return metric.compute()


def evaluate_recognition_model(model_name, model, image_infos, worker_num: int = 0, group_size: int = 1024000,
                               print_info: bool = False):
    """
    Evaluating recognition models using edit-distance algorithm.
    model can either implement [__call__] or [recognize] method
    """
    metric = RecognitionMetric()

    def recognition_postprocessor(args):
        if print_info: print("recognize ", args.detection_index, args.recognition_index, args.image_path)
        metric.append(args.pattern, args.text)

    evaluate_model(model_name=model_name, model=model, image_infos=image_infos, worker_num=worker_num,
                   group_size=group_size,
                   recognition_postprocessor=recognition_postprocessor)

    return metric.compute()


def evaluate_detection_and_recognition_model(model_name, model, image_infos, worker_num: int = 0,
                                             group_size: int = 1024000, print_info: bool = False):
    """
    Evaluating models that function as both detector and recognizer.
    model can either implement [__call__] or ([detect] & [recognize]) method
    """
    detection_metric = DetectionMetric()
    recognition_metric = RecognitionMetric()

    def detection_postprocessor(args):
        if print_info: print("detect ", args.detection_index, args.image_path)
        detection_metric.append([bbox for (text, bbox, confidence) in args.gts],
                                [(bbox, confidence) for text, bbox, confidence in args.result])

    def recognition_postprocessor(args):
        if print_info: print("recognize ", args.detection_index, args.recognition_index, args.image_path)
        recognition_metric.append(args.pattern, args.text)

    evaluate_model(model_name=model_name, model=model, image_infos=image_infos, worker_num=worker_num,
                   group_size=group_size,
                   detection_postprocessor=detection_postprocessor,
                   recognition_postprocessor=recognition_postprocessor)

    return detection_metric.compute(), recognition_metric.compute()


def filter_images(threshold: float, filtered_index_file_path: str):
    from difflib import SequenceMatcher

    gicp_results = ImageResultManager('result_recognition_gicp.json', 'sub_images')
    teg_results = ImageResultManager('result_recognition_teg.json', 'sub_images')
    output_results = []
    cnt = 0
    total_cnt = 0

    for image_path, gicp_value in gicp_results.__iter__(require_image=False):
        if not os.path.exists(image_path):
            continue
        teg_value = teg_results.get(os.path.basename(image_path), require_image=False)
        gicp_text = gicp_value.get('pattern') if gicp_value is not None else None
        gicp_text = gicp_text if gicp_text is not None else ""
        teg_text = teg_value.get('pattern') if teg_value is not None else None
        teg_text = teg_text if teg_text is not None else ""

        # print("matching: ", gicp_text, teg_text)
        ratio = SequenceMatcher(None, gicp_text, teg_text).ratio()
        if ratio <= threshold:
            output_results.append((image_path, gicp_text, teg_text, ratio))
            if cnt % 1000 == 0: print(total_cnt, cnt, image_path, gicp_text, teg_text, ratio)
            cnt += 1
        total_cnt += 1

    with open(filtered_index_file_path, "w", encoding="utf8") as f:
        json.dump(output_results, f, ensure_ascii=False)


if __name__ == "__main__":
    import time
    import json
    from difflib import SequenceMatcher

    start = time.time()

    # model = CompositeModel(YoloOpenvinoDetector(), PaddleOpenvinoRecognizer())
    model = TegRecognizer()
    image_infos = load_example_image_infos()
    metric = DetectionMetric()
    metric2 = RecognitionMetric()

    # model = NoRecognizer()
    # image_infos = json.load(open("teg_image_infos.json", "r", encoding="utf8"))
    # json.dump(create_image_infos("ocr_410_anno/testsets", model, worker_num=16), open("teg_image_infos.json", "w", encoding="utf8"), ensure_ascii=False, indent=4)
    # exit(0)

    def detection_postprocessor(args):
        metric.append([bbox for _, bbox, _ in args.gts], [(bbox, confidence) for _, bbox, confidence in args.result])
        image = make_boxed_image(args.image, [bbox for _, bbox, _ in args.result])
        image_name = os.path.basename(args.image_path)
        write_image_path = os.path.join(create_dir("teg_detection_images"), image_name)
        cv2.imwrite(write_image_path, image)
        print(args.image_path)

    def recognition_postprocessor(args):
        metric2.append(args.pattern, args.text)
        image = args.sub_image
        # image = make_boxed_image(image, [bbox for _, bbox, _ in model.detect(args.sub_image)])
        image = make_texted_image(image, ["pattern: " + args.pattern, "text: " + args.text])
        image_name = make_sub_image_name(os.path.basename(args.image_path), args.bbox)
        write_image_path = os.path.join(create_dir("teg_recognition_raw_images"), image_name)
        cv2.imwrite(write_image_path, image)

    evaluate_model("", model, image_infos, detection_postprocessor=detection_postprocessor, recognition_postprocessor=None, group_size=16, worker_num=32)
    print("score: ", metric.compute(), metric2.compute())

    print("total_time: ", time.time() - start)

    image_infos = create_image_infos("", model1)

    def recognize_processor(args):
        model2....

    evaluate_model("", model2, image_infos, recognition_postprocessor=recognize_processor)
