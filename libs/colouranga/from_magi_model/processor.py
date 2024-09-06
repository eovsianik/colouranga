# flake8: noqa: TRY003, TRY004

import numpy as np
import torch

# from shapely.geometry import box
from libs.colouranga.from_magi_model.config import MagiConfig
from libs.colouranga.from_magi_model.utils import x1y1x2y2_to_xywh
from transformers import ConditionalDetrImageProcessor, ViTImageProcessor

# from shapely.geometry import box


class MagiProcessor:
    def __init__(self, config: MagiConfig):
        self.config = config
        self.detection_image_preprocessor = ConditionalDetrImageProcessor.from_dict(
            config.detection_image_preprocessing_config
        )

        self.crop_embedding_image_preprocessor = ViTImageProcessor.from_dict(
            config.crop_embedding_image_preprocessing_config
        )
        self.margin = 10

    def preprocess_inputs_for_detection(self, images, annotations=None):
        images = list(images)  # list len: 5
        assert isinstance(images[0], np.ndarray)
        annotations = self._convert_annotations_to_coco_format(annotations)
        inputs = self.detection_image_preprocessor(
            images, annotations=annotations, return_tensors="pt"
        )  # type: ignore
        return inputs

    def preprocess_inputs_for_crop_embeddings(self, images):
        images = list(images)

        return self.crop_embedding_image_preprocessor(images, return_tensors="pt").pixel_values  # type: ignore

    def crop_image(self, image, bboxes):
        crops_for_image = []
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox

            # fix the bounding box in case it is out of bounds or too small
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            x1, y1, x2, y2 = min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)  # just incase
            x1, y1 = max(0, x1), max(0, y1)
            x1, y1 = min(image.shape[1], x1), min(image.shape[0], y1)
            x2, y2 = max(0, x2), max(0, y2)
            x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)

            if x2 - x1 < self.margin:
                if image.shape[1] - x1 > self.margin:
                    x2 = x1 + self.margin
                else:
                    x1 = x2 - self.margin
            if y2 - y1 < self.margin:
                if image.shape[0] - y1 > self.margin:
                    y2 = y1 + self.margin
                else:
                    y1 = y2 - self.margin

            crop = image[y1:y2, x1:x2]

            crops_for_image.append(crop)

        return crops_for_image

    def crop_image_new(self, image, bboxes):
        crops_for_image = {}
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox

            # fix the bounding box in case it is out of bounds or too small
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            x1, y1, x2, y2 = min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)  # just incase
            x1, y1 = max(0, x1), max(0, y1)
            x1, y1 = min(image.shape[1], x1), min(image.shape[0], y1)
            x2, y2 = max(0, x2), max(0, y2)
            x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)

            if x2 - x1 < self.margin:
                if image.shape[1] - x1 > self.margin:
                    x2 = x1 + self.margin
                else:
                    x1 = x2 - self.margin
            if y2 - y1 < self.margin:
                if image.shape[0] - y1 > self.margin:
                    y2 = y1 + self.margin
                else:
                    y1 = y2 - self.margin

            crop = image[y1:y2, x1:x2]

            crops_for_image[bbox] = crop

        return crops_for_image

    def _get_indices_of_characters_to_keep(
        self, batch_scores, batch_labels, batch_bboxes, character_detection_threshold
    ):
        indices_of_characters_to_keep = []
        for scores, labels, _ in zip(batch_scores, batch_labels, batch_bboxes, strict=False):
            indices = torch.where((labels == 0) & (scores > character_detection_threshold))[0]

            indices_of_characters_to_keep.append(indices)
        return indices_of_characters_to_keep

    def _convert_annotations_to_coco_format(self, annotations):
        if annotations is None:
            return None
        self._verify_annotations_are_in_correct_format(annotations)
        coco_annotations = []
        for annotation in annotations:
            coco_annotation = {
                "image_id": annotation["image_id"],
                "annotations": [],
            }

            for bbox, label in zip(
                annotation["bboxes_as_x1y1x2y2"], annotation["labels"], strict=False
            ):
                coco_annotation["annotations"].append(
                    {
                        "bbox": x1y1x2y2_to_xywh(bbox),
                        "category_id": label,
                        "area": (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]),
                    }
                )
            coco_annotations.append(coco_annotation)
        return coco_annotations

    def _verify_annotations_are_in_correct_format(self, annotations):
        error_msg = """
        Annotations must be in the following format:
        [
            {
                "image_id": 0,
                "bboxes_as_x1y1x2y2": [[0, 0, 10, 10], [10, 10, 20, 20], [20, 20, 30, 30]],
                "labels": [0, 1, 2],
            },
            ...
        ]
        Labels: 0 for characters, 1 for text, 2 for panels.
        """
        if annotations is None:
            return
        if not isinstance(annotations, list) and not isinstance(annotations, tuple):
            raise ValueError(f"{error_msg} Expected a List/Tuple, found {type(annotations)}.")
        if len(annotations) == 0:
            return
        if not isinstance(annotations[0], dict):
            raise ValueError(f"{error_msg} Expected a List[Dict], found {type(annotations[0])}.")
        if "image_id" not in annotations[0]:
            raise ValueError(f"{error_msg} Dict must contain 'image_id'.")
        if "bboxes_as_x1y1x2y2" not in annotations[0]:
            raise ValueError(f"{error_msg} Dict must contain 'bboxes_as_x1y1x2y2'.")
        if "labels" not in annotations[0]:
            raise ValueError(f"{error_msg} Dict must contain 'labels'.")
