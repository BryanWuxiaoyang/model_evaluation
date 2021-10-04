# coding=utf-8
from difflib import SequenceMatcher
import base64
import requests
import cv2
import l5sys


# 调用OCR服务对image进行文字识别
# 目前支持TEG数平OCR服务和游戏说GICP OCR服务
def recognize_ocr(image, method='teg'):
    text = None
    if image is None or (method != 'teg' and method != 'gicp'):
        return text

    _, binary_image = cv2.imencode('.png', image)

    if method == 'teg':
        _, qos = l5sys.ApiGetRoute({'modId': 64215233, 'cmdId': 655360}, 1.0)
        # prepare request payload
        #################### old version #####################
        data = {
            "interface": 5,
            "business": "ieg_game_recommend_str"
        }
        ######################################################

        #################### new version #####################
        # data = {
        #     "interface": 2802,
        #     "business": "teg_ocr"
        # }
        ######################################################

        files = {
            "myfile": binary_image
        }
    elif method == 'gicp':
        _, qos = l5sys.ApiGetRoute({'modId': 64987585, 'cmdId': 720896}, 1.0)
        # headers = {'x-odp-destination-service':'gametalk-ocr-system',
        #             'x-odp-source-service': 'go_test',
        #             'scene': 'detect',
        #             'host': 'gametalk-ocr-system'}

        ########################## old version #################################
        headers = {
            'x-odp-destination-service': 'gametalk-ocr-system',
            'x-odp-source-service': 'go_test',
            'scene': 'dnf_daoju',
            'host': 'gametalk-ocr-system'
        }
        #########################################################################
        img_data = base64.b64encode(binary_image)
    
    _model_server_addr = "http://{}:{}".format(qos["hostIp"], qos["hostPort"])

    # call model server
    try:
        if method == 'teg':
            response = requests.post(url=_model_server_addr, data=data, files=files)
            # deal with return content from server
            result = response.json()
            if result["errCode"] == 0:
                contents = result["contents"]
                if contents['error_code'] == 0:
                    text = contents['total_lines'][0]["line_sets"]
        elif method == 'gicp':
            response = requests.post(url=_model_server_addr, data=img_data, headers=headers)
            result = response.json()
            if result['errno'] == 0:
                text = result['res']

    except requests.ConnectTimeout:
        print("connection timeout")
        text = recognize_ocr(image, method)
        # raise RuntimeError()
    except requests.ConnectionError:
        print("connection error")
        text = recognize_ocr(image, method)
        # raise RuntimeError()
    except:
        print("ocr error")
        text = None

    return text


# 计算两个字符串的匹配度
# max_match_ratio为两个字符串的最大匹配度；offset为最大匹配度时字串的偏移量
def match_text(text_origin, text_match):
    # text_origin：原图中的字符串
    # text_match：待匹配的模板字符串
    len_origin = len(text_origin)
    len_match = len(text_match)

    if len_match > len_origin:
        return 0, 0

    max_match_ratio = 0
    offset = 0
    for idx in range(0, len_origin - len_match + 1):
        match_ratio = SequenceMatcher(None, text_origin[idx:idx+len_match], text_match).ratio()
        if match_ratio > max_match_ratio:
            max_match_ratio = match_ratio
            offset = idx

    return max_match_ratio, offset


# 获取一次识别的所有文字及坐标列表
def recognize_text(image, method='teg'):
    result_text = []
    result_coord = []

    ocr_results = recognize_ocr(image, method)
    if ocr_results is not None:
        for content in ocr_results:
            text = content['line_string']
            coord = content['line_box']['coordinate']
            result_text.append(text)
            result_coord.append(coord)
    
    return result_text, result_coord


if __name__ == '__main__':
    ratio = match_text('城无不胜上衣', '战无不胜上衣')
    # ratio = SequenceMatcher(None, '强者啊引导时间的流向，新杀敌人吧！', '强者啊，引导时间的流向，斩杀敌人吧！').ratio()
    print(ratio)
