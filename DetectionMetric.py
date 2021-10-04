import numpy as np


class DetectionMetric:
    """
    Evaluating detection model by MAP algorithm
    """
    def __init__(self):
        self.__detection_elements = []

    def append(self, ground_truths: list, preds: list):
        """
        :param ground_truths: list<bbox>, A set of ground truth bounding boxes(represented as 8-length list)
        :param preds: list<(bbox, confidence)>, A set of predictions of bounding boxes
        """
        self.__detection_elements.append((ground_truths, preds))

    def compute(self) -> float:
        """
        :return: Result of evaluation
        """
        if len(self.__detection_elements) == 0:
            return 0.0

        from mean_average_precision.metric_builder import MetricBuilder
        detection_elements = self.__detection_elements
        metric_fn = MetricBuilder.build_evaluation_metric("map_2d", async_mode=False, num_classes=1)
        for gts, preds in detection_elements:
            new_gts = []
            for gt in gts:
                new_gts.append([gt[0], gt[1], gt[2], gt[3], 0, 0, 0, gt[4], gt[5], gt[6], gt[7]])

            new_preds = []
            for pred, confidence in preds:
                new_preds.append([pred[0], pred[1], pred[2], pred[3], 0, confidence, pred[4], pred[5], pred[6], pred[7]])

            new_gts = np.asarray(new_gts)
            new_preds = np.asarray(new_preds)
            # print("new_gts: ", new_gts)
            # print("new_preds: ", new_preds)
            metric_fn.add(new_preds, new_gts)

        return float(metric_fn.value(iou_thresholds=0.5)['mAP'])
