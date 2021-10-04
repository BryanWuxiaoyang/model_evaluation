import numpy as np
from utils import *


class NoRecognizer:
    def __call__(self, image: np.ndarray):
        return []


class GicpRecognizer:
    def __init__(self):
        pass

    def __call__(self, image: np.ndarray):
        import ocr
        ocr_results = ocr.recognize_ocr(image, 'gicp')
        ret = []
        if ocr_results is not None:
            for result in ocr_results:
                text = result['line_string']
                confidence = result['line_confidence']
                points = result['line_box']['coordinate']
                ret.append((text, points, confidence))
        return ret

    def detect(self, image: np.ndarray):
        import ocr
        ocr_results = ocr.recognize_ocr(image, 'gicp')
        ret = []
        if ocr_results is not None:
            for result in ocr_results:
                # text = result['line_string']
                confidence = result['line_confidence']
                points = result['line_box']['coordinate']
                ret.append(("", points, confidence))
        return ret

    def recognize(self, image: np.ndarray):
        import ocr
        ocr_results = ocr.recognize_ocr(image, 'gicp')
        ret = []
        if ocr_results is not None:
            for result in ocr_results:
                text = result['line_string']
                confidence = result['line_confidence']
                points = result['line_box']['coordinate']
                ret.append((text, points, confidence))
        return ret


class TegRecognizer:
    def __call__(self, image: np.ndarray):
        import ocr
        ocr_results = ocr.recognize_ocr(image, 'teg')
        ret = []
        if ocr_results is not None:
            for result in ocr_results:
                text = result['line_string']
                confidence = result['line_confidence']
                points = result['line_box']['coordinate']
                ret.append((text, points, confidence))
        return ret


class YoutuDetector:
    def __init__(self):
        from detection_youtu.detector import Detector
        self.detector = Detector("detection_youtu/PTD-2021-0531.pth.tar")

    def __call__(self, image: np.ndarray):
        ocr_results = self.detector.inference(image, False)
        ret = []
        if ocr_results is not None:
            for bbox in ocr_results:
                text = ""
                confidence = 1.0
                points = [bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1],
                          bbox[2][0], bbox[2][1], bbox[3][0], bbox[3][1]]
                ret.append((text, points, confidence))
        # print("ocr result: ", ret)
        return ret


class PaddleOcrRecognizer:
    def __init__(self, det_model_dir: str = None, rec_model_dir: str = None):
        import paddleocr
        self.det_model_dir = det_model_dir
        self.rec_model_dir = rec_model_dir
        self.model = paddleocr.PaddleOCR(det_model_dir=det_model_dir, rec_model_dir=rec_model_dir,
                               use_angle_cls=False, lang='ch')

    def __call__(self, image: np.ndarray):
        ocr_results = self.model.ocr(image, cls=False, det=self.det_model_dir is not None, rec=self.rec_model_dir is not None)
        ret = []
        if self.rec_model_dir is None:
            for points in ocr_results:
                points = flatten_box(points)
                ret.append(("", points, 1.0))
        elif self.det_model_dir is None:
            ret.append(("", ocr_results[0], ocr_results[1]))
        else:
            for (points, (text, confidence)) in ocr_results:
                points = flatten_box(points)
                ret.append((text, points, confidence))
        return ret

    def detect(self, image: np.ndarray):
        ocr_results = self.model.ocr(image, cls=False, det=True, rec=False)
        ret = []
        for points in ocr_results:
            points = flatten_box(points)
            ret.append(("", points, 1.0))
        return ret

    def recognize(self, image: np.ndarray):
        ocr_results = self.model.ocr(image, cls=False, det=False, rec=True)
        ret = []
        for text, confidence in ocr_results:
            ret.append((text, None, confidence))
        return ret


class TechFrameDetector:
    def __init__(self):
        from detection_techframe.test import generate_model
        config_path = "detection_techframe/configs/deploy.conf"
        model_path = "detection_techframe/models/model.pth"
        self.detector = generate_model(config_path, model_path)

    def __call__(self, image: np.ndarray):
        from utils import flatten_box
        import cv2
        import PIL.Image
        image = PIL.Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ocr_results = self.detector.apply_img(image)
        ret = []  # list<(text, bbox, confidence)>
        # print("ocr_results: ", ocr_results)
        boxes, confidences = ocr_results
        for box, confidence in zip(boxes, confidences):
            text = ""
            points = flatten_box(box)
            ret.append((text, points, confidence))
        # print("ret: ", ret)
        return ret


class TegHighPrecisionRecognizer:
    def __init__(self):
        pass

    def __call__(self, image: np.ndarray):
        if image is None: return []

        return self.detect(image)

    def detect(self, image: np.ndarray):
        if image is None: return []

        result = self.get_result_by_img(image, 0)
        return self.postprocess(result)

    def recognize(self, image: np.ndarray):
        if image is None: return []

        result = self.get_result_by_img(image, 1)
        return self.postprocess(result)

    def postprocess(self, result):
        ret = []
        if result is not None and isinstance(result, dict) and result.get('response') is not None and \
                len(result['response']) != 0 and \
                isinstance(result['response'][0], dict) and result['response'][0].get('paragraph') is not None:
            for attr_map in result['response'][0]['paragraph']:
                for line_map in attr_map['lines']:
                    confidence = line_map['score']
                    text = line_map['line']
                    box = line_map['lineBox']
                    ret.append((text, box, confidence))
        return ret

    def get_ip_port_url(self, modId, cmdId):
        import l5sys
        data = {
            'modId': modId,
            'cmdId': cmdId,
        }
        ret, qos = l5sys.ApiGetRoute(data, 0.2)
        ip = qos['hostIp']
        port = qos['hostPort']
        url = 'http://{:s}:{:d}'.format(ip, port)

        return ip, port, url

    def get_result_by_img(self, img, recog_only):
        import base64
        import json
        import http.client

        try:
            IP, PORT, URL = self.get_ip_port_url(192000065, 78441)

            data = {
                'interface': '2802',  # 接口ID
                'business': 'teg_ocr',  # 业务ID
                'recog_only': recog_only,  # 0/1,只识别词条则该参数值填1，默认值为0
                'img_data': base64.b64encode(cv2.imencode('.jpg', img)[1].tobytes()).decode('utf-8')
            }
            data = json.dumps(data)

            headerdata = {"Content-type": "json"}
            conn = http.client.HTTPConnection(IP, PORT)
            conn.request('POST', URL, data, headerdata)
            response = conn.getresponse()
            result = response.read()

            return json.loads(result) if result is not None else None
        except http.client.REQUEST_TIMEOUT:
            print("connection timeout")
            return self.get_result_by_img(img, recog_only)
        except BrokenPipeError:
            print("broken pipe error")
            return self.get_result_by_img(img, recog_only)
        except:
            print("ocr error")
            return self.get_result_by_img(img, recog_only)


class PaddleOnnxRecognizer:
    def __init__(self):
        from PaddleOCR_onnx.onnx_inference.predict_rec import MyTextRecognizer
        self.model = MyTextRecognizer()

    def __call__(self, image: np.ndarray):
        return self.recognize(image)

    def recognize(self, image: np.ndarray):
        results = self.model(image)
        ret = [(text, None, confidence) for text, confidence in results]
        return ret


class PaddleOpenvinoRecognizer:
    def __init__(self):
        from PaddleOCR_onnx.onnx_inference.predict_rec import OpenvinoTextRecognizer
        self.model = OpenvinoTextRecognizer()

    def __call__(self, image: np.ndarray):
        return self.recognize(image)

    def recognize(self, image: np.ndarray):
        results = self.model(image)
        ret = [(text, None, confidence) for text, confidence in results]
        return ret


class YoloOnnxDetector:
    def __init__(self):
        from yolo_detector.YoloDetect import YoloDetect
        self.model = YoloDetect()

    def __call__(self, image: np.ndarray):
        return self.detect(image)

    def detect(self, image: np.ndarray):
        result = self.model(image)
        return self.model(image)


class YoloOpenvinoDetector:
    def __init__(self):
        from yolo_detector.YoloDetect import YoloOpenvinoDetect
        self.model = YoloOpenvinoDetect()

    def __call__(self, image: np.ndarray):
        return self.detect(image)

    def detect(self, image: np.ndarray):
        return self.model(image)


class CompositeModel:
    def __init__(self, detection_model=None, recognition_model=None):
        self.detection_model = detection_model
        self.recognition_model = recognition_model

    def detect(self, image: np.ndarray):
        return self.detection_model.detect(image) if hasattr(self.detection_model, "detect") else self.detection_model(image)

    def recognize(self, image: np.ndarray):
        result = self.detection_model.detect(image) if hasattr(self.detection_model, "detect") else self.detection_model(image)

        ret = []
        from utils import make_sub_image

        for _, bbox, confidence in result:
            sub_image = make_sub_image(image, bbox)
            sub_result = self.recognition_model.recognize(sub_image) if hasattr(self.recognition_model, "recognize")\
                else self.recognition_model(sub_image)

            for text, _, _ in sub_result:
                ret.append((text, bbox, confidence))
        return ret


if __name__ == "__main__":
    model = PaddleOpenvinoRecognizer()
    import cv2
    image = cv2.imread("test.jpg", cv2.IMREAD_COLOR)
    result = model(image)
    print(result)
    pass
