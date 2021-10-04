from utils import *
from models import *
from AutoSave import *
from ResultManager import *
from model_evaluation import evaluate_model


def recognition_filter(gt_model, cmp_model, image_dir: str, write_image_dir: str, write_image_index_file: str,
                       threshold: float = 0.4):
    gt_image_infos = create_image_infos(image_dir, gt_model)

    manager = ImageResultManager(write_image_index_file, write_image_dir)

    def recognition_postprocessor(args):
        print(args.image_path)
        gt_text = args.text
        cmp_text = args.pattern
        ratio = args.ratio
        if ratio < threshold:
            manager.append(args.sub_image, make_sub_image_name(os.path.basename(args.image_path), args.bbox), gt_text=gt_text, cmp_text=cmp_text, ratio=ratio)

    evaluate_model("", cmp_model, gt_image_infos, recognition_postprocessor=recognition_postprocessor, group_size=16)
    manager.save()


def detection_filter(gt_model, cmp_model, image_dir: str, write_image_dir: str, write_image_index_file: str,
                     threshold: float = 0.4):
    gt_image_infos = create_image_infos(image_dir, gt_model,)

    manager = ImageResultManager(write_image_index_file, write_image_dir)

    gt_image_infos = AutoSaveIterator(
        iterator_generator=lambda iter_cnt: gt_image_infos[iter_cnt:].__iter__(),
        resume_iter_file_path="example_iter",
        loader=None,
        saver=lambda iter_cnt: manager.save(),
        save_every=1000,
        auto_iter=False,
        auto_save=True,
        restart=False,
        print_info=True,
    )

    def detection_postprocessor(args):
        print(args.image_path)
        gt_bbox = [bbox for _, bbox, _ in args.gts]
        cmp_preds = [(bbox, confidence) for _, bbox, confidence in args.result]

        from DetectionMetric import DetectionMetric
        metric = DetectionMetric()
        metric.append(gt_bbox, cmp_preds)

        ratio = metric.compute()
        if ratio < threshold:
            manager.append(args.image, os.path.basename(args.image_path), gt_result=args.gts, cmp_result=args.result, ratio=float(ratio))

        gt_image_infos.try_iter()

    evaluate_model("", cmp_model, gt_image_infos, detection_postprocessor=detection_postprocessor, group_size=16)
    manager.save()


if __name__ == "__main__":
    recognition_filter(TegRecognizer(), GicpRecognizer(), "ocr_410_anno/testsets", "filter_rec_images", "filter_rec_index_file.json")
    detection_filter(TegRecognizer(), GicpRecognizer(), "ocr_410_anno/testsets", "filter_det_images", "filter_det_index_file.json")
