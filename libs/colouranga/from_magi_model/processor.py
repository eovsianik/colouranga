from typing import List

import numpy as np
import torch

# from shapely.geometry import box
from libs.colouranga.from_magi_model.config import MagiConfig
from libs.colouranga.from_magi_model.utils import (
    UnionFind,
    sort_panels,
    sort_text_boxes_in_reading_order,
    x1y1x2y2_to_xywh,
)
from transformers import ConditionalDetrImageProcessor, ViTImageProcessor
from transformers.image_transforms import center_to_corners_format


class MagiProcessor:
    def __init__(self, config: MagiConfig):
        self.config = config
        self.detection_image_preprocessor = ConditionalDetrImageProcessor.from_dict(
            config.detection_image_preprocessing_config
        )

        self.crop_embedding_image_preprocessor = ViTImageProcessor.from_dict(
            config.crop_embedding_image_preprocessing_config
        )

    def preprocess_inputs_for_detection(self, images, annotations=None):  # images - some arrays
        images = list(images)  # list len: 5
        assert isinstance(images[0], np.ndarray)
        annotations = self._convert_annotations_to_coco_format(
            annotations
        )  # annotations - class 'NoneType' непонятно вообще зачем
        inputs = self.detection_image_preprocessor(
            images, annotations=annotations, return_tensors="pt"
        )  # type: ignore , inputs input <class 'transformers.feature_extraction_utils.BatchFeature'>
        return inputs

    def preprocess_inputs_for_crop_embeddings(self, images):
        images = list(images)
        # assert isinstance(images[0], np.ndarray)
        return self.crop_embedding_image_preprocessor(images, return_tensors="pt").pixel_values  # type: ignore
        # pixel_values (torch.FloatTensor of shape (batch_size, num_channels, height, width)) — Pixel values. Pixel values can be obtained using AutoImageProcessor. See ViTImageProcessor.call() for details.

    # def move_to_device(self, input):
    #     return move_to_device(input, self.device)

    def postprocess_detections_and_associations(
        self,
        images,
        predicted_bboxes,
        predicted_class_scores,
        original_image_sizes,  # [кол-во картинок, ширина и длина]
        get_character_character_matching_scores,
        character_detection_threshold=0.3,
        panel_detection_threshold=0.2,
        text_detection_threshold=0.25,
        character_character_matching_threshold=0.65,
    ):
        # Получение максимальные оценки и метки для каждого предсказания
        batch_scores, batch_labels = predicted_class_scores.max(-1)
        # batch_scores [кол-во картинок, 300]
        # batch_labels [кол-во картинок, 300]
        # Сигмоидная функция применяется к оценкам
        batch_scores = batch_scores.sigmoid()
        # batch_scores [кол-во картинок, 300]
        # Преобразование меток в целые числа
        batch_labels = batch_labels.long()
        # batch_labels [кол-во картинок, 300]
        # Преобразование предсказанных ограничивающих рамок
        batch_bboxes = center_to_corners_format(predicted_bboxes)  # type: ignore
        # batch_bboxes [кол-во картинок, 300, 4]

        # Масштабирование ограничивающих рамок обратно к размерам оригинального изображения
        if isinstance(original_image_sizes, List):
            img_h = torch.Tensor([i[0] for i in original_image_sizes])
            img_w = torch.Tensor([i[1] for i in original_image_sizes])
        else:
            img_h, img_w = original_image_sizes.unbind(1)
            # img_h, img_w у обоих [кол-во картинок]
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(batch_bboxes.device)
        batch_bboxes: torch.Tensor = batch_bboxes * scale_fct[:, None, :]

        # Получение индексов панелей, персонажей и текстов для сохранения
        batch_panel_indices = self._get_indices_of_panels_to_keep(
            batch_scores, batch_labels, batch_bboxes, panel_detection_threshold
        )  # list внутри list'ы, количество которых равно количеству картинок, во вложенных листах индексы панелей с персами
        batch_character_indices = self._get_indices_of_characters_to_keep(
            batch_scores, batch_labels, batch_bboxes, character_detection_threshold
        )  # list - длина равна количеству картинок,элементы - тензоры с индексами

        # нужна ли вообще эта строка????? DEL
        batch_text_indices = self._get_indices_of_texts_to_keep(
            batch_scores, batch_labels, batch_bboxes, text_detection_threshold
        )

        #! СЮДА НАДО КАК-ТО ВСТАВИТЬ
        # Получение оценок совпадения персонажей - тут MB надо менять, чтобы не по одной картинке шло
        batch_character_character_matching_scores = get_character_character_matching_scores(
            batch_character_indices, batch_bboxes
        )
        # batch_character_indices list - количество картинок
        # batch_bboxes torch.Tensor [кол-во картинок, 300, 4 координаты] - координаты всех боксов

        # Сортировка панелей и текстовых блоков в порядке чтения
        for batch_index in range(len(batch_scores)):
            panel_bboxes = batch_bboxes[batch_index][batch_panel_indices[batch_index]]
            panel_scores = batch_scores[batch_index][batch_panel_indices[batch_index]]
            text_bboxes = batch_bboxes[batch_index][batch_text_indices[batch_index]]
            text_scores = batch_scores[batch_index][batch_text_indices[batch_index]]

            sorted_panel_indices = sort_panels(panel_bboxes)
            # список отсортированных индексов панелей
            batch_bboxes[batch_index][batch_panel_indices[batch_index]] = panel_bboxes[  # type: ignore
                sorted_panel_indices
            ]
            # batch_bboxes [кол-во картинок, 300, 4] -- полагаю, что это координаты всех bbox
            batch_scores[batch_index][batch_panel_indices[batch_index]] = panel_scores[
                sorted_panel_indices
            ]
            # batch_scores [кол-во картинок, 300]
            sorted_panels = batch_bboxes[batch_index][batch_panel_indices[batch_index]]
            # sorted_panels [кол-во bbox на одной картинке, координаты]
            # МБ ВООБЩЕ УБРАТЬ? ТУТ ПРО ТЕКСТ
            sorted_text_indices = sort_text_boxes_in_reading_order(text_bboxes, sorted_panels)
            batch_bboxes[batch_index][batch_text_indices[batch_index]] = text_bboxes[  # type: ignore
                sorted_text_indices
            ]
            # короч, строчка выше к тексту отношение имеет - из всех bbox выбираем текстовые
            # да и строчка ниже, полагаю
            batch_scores[batch_index][batch_text_indices[batch_index]] = text_scores[
                sorted_text_indices
            ]

        results = []

        # тут по идее половину бы вырезать, но код будет выть
        # Ограничивающие рамки, оценки и индексы для панелей, текста и персонажей
        for batch_index in range(len(batch_scores)):
            panel_bboxes = batch_bboxes[batch_index][batch_panel_indices[batch_index]]
            panel_scores = batch_scores[batch_index][batch_panel_indices[batch_index]]
            text_bboxes = batch_bboxes[batch_index][batch_text_indices[batch_index]]
            text_scores = batch_scores[batch_index][batch_text_indices[batch_index]]
            character_bboxes = batch_bboxes[batch_index][batch_character_indices[batch_index]]
            # character_bboxes [кол-во bbox на странице, 4 - координаты]
            character_scores = batch_scores[batch_index][batch_character_indices[batch_index]]
            # character_bboxes [кол-во bbox на странице]
            # Ассоциации между персонажами на основе заданного порога
            char_i, char_j = torch.where(
                batch_character_character_matching_scores[batch_index]
                > character_character_matching_threshold
            )
            # char_i, char_j - размер [45] у обоих - не знаю источник данной цифры
            character_character_associations = torch.stack([char_i, char_j], dim=1)
            # character_character_associations [45, 2]
            # Для сопоставления персонажей
            character_ufds = UnionFind.from_adj_matrix(
                batch_character_character_matching_scores[batch_index]
                > character_character_matching_threshold
            )
            # character_ufds - class

            results.append(
                {
                    # "panels": panel_bboxes.tolist(),
                    # "panel_scores": panel_scores.tolist(),
                    # "texts": text_bboxes.tolist(),
                    # "text_scores": text_scores.tolist(),
                    "characters": character_bboxes.tolist(),
                    "character_scores": character_scores.tolist(),
                    "character_character_associations": character_character_associations.tolist(),
                    "character_cluster_labels": character_ufds.get_labels_for_connected_components(),
                }
            )
            # этот results - список по каждой странице
        return results

    # Обрезка изображения по заданным ограничивающим рамкам, корректируя их при необходимости, чтобы обеспечить корректный размер и положение обрезанных областей
    def crop_image(self, image, bboxes):
        crops_for_image = []
        for bbox in bboxes:
            # берут координаты каждой рамочки из каждой картинки
            x1, y1, x2, y2 = bbox
            # x1, y1, x2, y2 - 4 тензора

            # fix the bounding box in case it is out of bounds or too small
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            x1, y1, x2, y2 = min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)  # just incase
            x1, y1 = max(0, x1), max(0, y1)
            x1, y1 = min(image.shape[1], x1), min(image.shape[0], y1)
            x2, y2 = max(0, x2), max(0, y2)
            x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)

            # Проверка: не слишком ли мала рамка
            if x2 - x1 < 10:
                if image.shape[1] - x1 > 10:
                    x2 = x1 + 10
                else:
                    x1 = x2 - 10
            if y2 - y1 < 10:
                if image.shape[0] - y1 > 10:
                    y2 = y1 + 10
                else:
                    y1 = y2 - 10

            # Обрезка изображения по ограничивающей рамке
            crop = image[y1:y2, x1:x2]
            # crop - строка
            # image - numpy array - И ЭТО ВАЖНО
            crops_for_image.append(crop)
            # crops_for_image - список с numpy array
            # numpy arrays размерами - [длина, ширина, 3 (каналы rgb)]
        return crops_for_image

    def crop_image_new(self, image, bboxes):
        crops_for_image = {}
        for bbox in bboxes:
            # берут координаты каждой рамочки из каждой картинки
            x1, y1, x2, y2 = bbox
            # x1, y1, x2, y2 - 4 тензора

            # fix the bounding box in case it is out of bounds or too small
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            x1, y1, x2, y2 = min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)  # just incase
            x1, y1 = max(0, x1), max(0, y1)
            x1, y1 = min(image.shape[1], x1), min(image.shape[0], y1)
            x2, y2 = max(0, x2), max(0, y2)
            x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)

            # Проверка: не слишком ли мала рамка
            if x2 - x1 < 10:
                if image.shape[1] - x1 > 10:
                    x2 = x1 + 10
                else:
                    x1 = x2 - 10
            if y2 - y1 < 10:
                if image.shape[0] - y1 > 10:
                    y2 = y1 + 10
                else:
                    y1 = y2 - 10

            # Обрезка изображения по ограничивающей рамке
            crop = image[y1:y2, x1:x2]
            # crop - строка
            # image - numpy array - И ЭТО ВАЖНО
            crops_for_image[bbox] = crop
            # crops_for_image - список с numpy array
            # numpy arrays размерами - [длина, ширина, 3 (каналы rgb)]
        return crops_for_image

    # Список для хранения индексов персонажей - в эту функцию не заходится upd: заходится, но во второй раз - зачем в первый - хз
    def _get_indices_of_characters_to_keep(
        self, batch_scores, batch_labels, batch_bboxes, character_detection_threshold
    ):  # batch_scores, batch_labels [кол-во картинок, 300], batch_bboxes [кол-во картинок, 300, 4], character_detection_threshold = 0,3
        indices_of_characters_to_keep = []
        for scores, labels, _ in zip(batch_scores, batch_labels, batch_bboxes):
            indices = torch.where((labels == 0) & (scores > character_detection_threshold))[0]
            # indices - [каждый раз разное число, которое равно количеству распознных персов, включая и их повторение]
            indices_of_characters_to_keep.append(indices)
        return indices_of_characters_to_keep  # list - длина равна количеству картинок

    # Список для хранения индексов панелей
    def _get_indices_of_panels_to_keep(
        self, batch_scores, batch_labels, batch_bboxes, panel_detection_threshold
    ):
        indices_of_panels_to_keep = []

        for scores, labels, bboxes in zip(batch_scores, batch_labels, batch_bboxes):
            # scores - [69], labels - [69], bboxes - [69, 4] - 69 - число боксов разное, 4 - координаты
            indices = torch.where(labels == 2)[0]
            # indices - [69]
            bboxes = bboxes[
                indices
            ]  # tuple - len 69 тензоров - тут по 4 координаты для каждого бокса
            scores = scores[indices]  # tuple - len 69 тензоров - тут вероятность - что это перс
            labels = labels[indices]  # tuple - len 69 тензоров для персонажей label = 2
            if len(indices) == 0:
                indices_of_panels_to_keep.append([])
                continue
            scores, labels, indices, bboxes = zip(
                *sorted(zip(scores, labels, indices, bboxes), reverse=True)
            )
            panels_to_keep = []
            union_of_panels_so_far = box(0, 0, 0, 0)  # class Polygon

            for ps, pb, pl, pi in zip(scores, bboxes, labels, indices):
                panel_polygon = box(pb[0], pb[1], pb[2], pb[3])
                # Пропуск панели с оценкой ниже заданного порога
                if ps < panel_detection_threshold:
                    continue
                # Пропуск панели, если их пересечение с уже найденными панелями больше 50% их площади
                if (
                    union_of_panels_so_far.intersection(panel_polygon).area / panel_polygon.area
                    > 0.5
                ):
                    continue
                panels_to_keep.append((ps, pl, pb, pi))  # list - len - количество картинок
                union_of_panels_so_far = union_of_panels_so_far.union(panel_polygon)  # polygon
            indices_of_panels_to_keep.append([p[3].item() for p in panels_to_keep])
        return indices_of_panels_to_keep  # list

    # Cписок для хранения индексов текстовых блоков - по делу удалить нафиг DEL
    def _get_indices_of_texts_to_keep(
        self, batch_scores, batch_labels, batch_bboxes, text_detection_threshold
    ):
        indices_of_texts_to_keep = []
        for scores, labels, bboxes in zip(batch_scores, batch_labels, batch_bboxes):
            indices = torch.where((labels == 1) & (scores > text_detection_threshold))[0]

            # Фильтрация ограничивающих рамок и оценки по найденным индексам
            bboxes = bboxes[indices]
            scores = scores[indices]
            labels = labels[indices]
            if len(indices) == 0:
                indices_of_texts_to_keep.append([])
                continue
            scores, labels, indices, bboxes = zip(
                *sorted(zip(scores, labels, indices, bboxes), reverse=True)
            )
            texts_to_keep = []
            texts_to_keep_as_shapely_objects = []

            for ts, tb, tl, ti in zip(scores, bboxes, labels, indices):
                text_polygon = box(tb[0], tb[1], tb[2], tb[3])
                should_append = True

                # Проверка пересечение с уже найденными текстовыми блоками
                for t in texts_to_keep_as_shapely_objects:
                    if t.intersection(text_polygon).area / t.union(text_polygon).area > 0.5:
                        should_append = False
                        break

                # Если текстовый блок не пересекается с другими, то происходит добавление его в список
                if should_append:
                    texts_to_keep.append((ts, tl, tb, ti))
                    texts_to_keep_as_shapely_objects.append(text_polygon)
            indices_of_texts_to_keep.append([t[3].item() for t in texts_to_keep])
        return indices_of_texts_to_keep

    # Конвертация аннотаций в формат Coco
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

            # Проход по каждой ограничивающей рамке и метке
            for bbox, label in zip(annotation["bboxes_as_x1y1x2y2"], annotation["labels"]):
                coco_annotation["annotations"].append(
                    {
                        "bbox": x1y1x2y2_to_xywh(bbox),
                        "category_id": label,
                        "area": (bbox[2] - bbox[0])
                        * (bbox[3] - bbox[1]),  # Площадь ограничивающей рамки
                    }
                )
            coco_annotations.append(coco_annotation)
        return coco_annotations

    # Проверка на правильный формат у аннотаций
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
        if not isinstance(annotations, List) and not isinstance(annotations, tuple):
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
