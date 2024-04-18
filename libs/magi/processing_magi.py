from transformers import ConditionalDetrImageProcessor, TrOCRProcessor, ViTImageProcessor
from transformers.image_transforms import center_to_corners_format
import torch
from typing import List
from shapely.geometry import box
from .utils import UnionFind, sort_panels, sort_text_boxes_in_reading_order, x1y1x2y2_to_xywh
import numpy as np

class MagiProcessor():
    def __init__(self, config):
        self.config = config
        self.detection_image_preprocessor = None
        self.ocr_preprocessor = None
        self.crop_embedding_image_preprocessor = None
        if not config.disable_detections:
            assert config.detection_image_preprocessing_config is not None
            self.detection_image_preprocessor =  ConditionalDetrImageProcessor.from_dict(config.detection_image_preprocessing_config)
        if not config.disable_ocr:
            assert config.ocr_pretrained_processor_path is not None
            self.ocr_preprocessor = TrOCRProcessor.from_pretrained(config.ocr_pretrained_processor_path)
        if not config.disable_crop_embeddings:
            assert config.crop_embedding_image_preprocessing_config is not None
            self.crop_embedding_image_preprocessor = ViTImageProcessor.from_dict(config.crop_embedding_image_preprocessing_config)
    
    def preprocess_inputs_for_detection(self, images, annotations=None):
        images = list(images)
        assert isinstance(images[0], np.ndarray)
        annotations = self._convert_annotations_to_coco_format(annotations)
        inputs = self.detection_image_preprocessor(images, annotations=annotations, return_tensors="pt")
        return inputs

    def preprocess_inputs_for_ocr(self, images):
        images = list(images)
        assert isinstance(images[0], np.ndarray)
        return self.ocr_preprocessor(images, return_tensors="pt").pixel_values
    
    def preprocess_inputs_for_crop_embeddings(self, images):
        images = list(images)
        assert isinstance(images[0], np.ndarray)
        return self.crop_embedding_image_preprocessor(images, return_tensors="pt").pixel_values
    
    def postprocess_detections_and_associations(
            self,
            predicted_bboxes,
            predicted_class_scores,
            original_image_sizes,
            get_character_character_matching_scores,
            get_text_character_matching_scores,
            get_dialog_confidence_scores,
            character_detection_threshold=0.3,
            panel_detection_threshold=0.2,
            text_detection_threshold=0.25,
            character_character_matching_threshold=0.65,
            text_character_matching_threshold=0.4,
        ):

        assert self.config.disable_detections is False
        # Получение максимальные оценки и метки для каждого предсказания
        batch_scores, batch_labels = predicted_class_scores.max(-1)
        # Сигмоидная функция применяется к оценкам
        batch_scores = batch_scores.sigmoid()
        # Преобразование меток в целые числа
        batch_labels = batch_labels.long()
        # Преобразование предсказанных ограничивающих рамок
        batch_bboxes = center_to_corners_format(predicted_bboxes)

        # Масштабирование ограничивающих рамок обратно к размерам оригинального изображения
        if isinstance(original_image_sizes, List):
            img_h = torch.Tensor([i[0] for i in original_image_sizes])
            img_w = torch.Tensor([i[1] for i in original_image_sizes])
        else:
            img_h, img_w = original_image_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(batch_bboxes.device)
        batch_bboxes = batch_bboxes * scale_fct[:, None, :]
        
        # Получение индексов панелей, персонажей и текстов для сохранения
        batch_panel_indices = self._get_indices_of_panels_to_keep(batch_scores, batch_labels, batch_bboxes, panel_detection_threshold)
        batch_character_indices = self._get_indices_of_characters_to_keep(batch_scores, batch_labels, batch_bboxes, character_detection_threshold)
        batch_text_indices = self._get_indices_of_texts_to_keep(batch_scores, batch_labels, batch_bboxes, text_detection_threshold)

        # Получение оценок совпадения персонажей и текста
        batch_character_character_matching_scores = get_character_character_matching_scores(batch_character_indices, batch_bboxes)
        batch_text_character_matching_scores = get_text_character_matching_scores(batch_text_indices, batch_character_indices)
        batch_dialog_confidence_scores = get_dialog_confidence_scores(batch_text_indices)

        # Сортировка панелей и текстовых блоков в порядке чтения
        for batch_index in range(len(batch_scores)):
            panel_bboxes = batch_bboxes[batch_index][batch_panel_indices[batch_index]]
            panel_scores = batch_scores[batch_index][batch_panel_indices[batch_index]]
            text_bboxes = batch_bboxes[batch_index][batch_text_indices[batch_index]]
            text_scores = batch_scores[batch_index][batch_text_indices[batch_index]]

            sorted_panel_indices = sort_panels(panel_bboxes)
            batch_bboxes[batch_index][batch_panel_indices[batch_index]] = panel_bboxes[sorted_panel_indices]
            batch_scores[batch_index][batch_panel_indices[batch_index]] = panel_scores[sorted_panel_indices]
            sorted_panels = batch_bboxes[batch_index][batch_panel_indices[batch_index]]

            sorted_text_indices = sort_text_boxes_in_reading_order(text_bboxes, sorted_panels)
            batch_bboxes[batch_index][batch_text_indices[batch_index]] = text_bboxes[sorted_text_indices]
            batch_scores[batch_index][batch_text_indices[batch_index]] = text_scores[sorted_text_indices]
            batch_text_character_matching_scores[batch_index] = batch_text_character_matching_scores[batch_index][sorted_text_indices]
            batch_dialog_confidence_scores[batch_index] = batch_dialog_confidence_scores[batch_index][sorted_text_indices]

        results = []

        # Ограничивающие рамки, оценки и индексы для панелей, текста и персонажей 
        for batch_index in range(len(batch_scores)):
            panel_bboxes = batch_bboxes[batch_index][batch_panel_indices[batch_index]]
            panel_scores = batch_scores[batch_index][batch_panel_indices[batch_index]]
            text_bboxes = batch_bboxes[batch_index][batch_text_indices[batch_index]]
            text_scores = batch_scores[batch_index][batch_text_indices[batch_index]]
            character_bboxes = batch_bboxes[batch_index][batch_character_indices[batch_index]]
            character_scores = batch_scores[batch_index][batch_character_indices[batch_index]]

            #Ассоциации между персонажами на основе заданного порога
            char_i, char_j = torch.where(batch_character_character_matching_scores[batch_index] > character_character_matching_threshold)
            character_character_associations = torch.stack([char_i, char_j], dim=1)

            # Сопоставление текстовых блоков с персонажами
            text_boxes_to_match = batch_dialog_confidence_scores[batch_index] > text_character_matching_threshold

            # Если нет текстовых блоков для сопоставления, создаем пустой список ассоциаций
            if 0 in batch_text_character_matching_scores[batch_index].shape:
                text_character_associations = torch.zeros((0, 2), dtype=torch.long)
            else:

                # Наиболее вероятный говорящий для каждого текстового блока
                most_likely_speaker_for_each_text = torch.argmax(batch_text_character_matching_scores[batch_index], dim=1)[text_boxes_to_match]
                text_indices = torch.arange(len(text_bboxes)).type_as(most_likely_speaker_for_each_text)[text_boxes_to_match]
                text_character_associations = torch.stack([text_indices, most_likely_speaker_for_each_text], dim=1)
            
            # Для сопоставления персонажей
            character_ufds = UnionFind.from_adj_matrix(
                batch_character_character_matching_scores[batch_index] > character_character_matching_threshold
            )
            results.append({
                "panels": panel_bboxes.tolist(),
                "panel_scores": panel_scores.tolist(),
                "texts": text_bboxes.tolist(),
                "text_scores": text_scores.tolist(),
                "characters": character_bboxes.tolist(),
                "character_scores": character_scores.tolist(),
                "character_character_associations": character_character_associations.tolist(),
                "text_character_associations": text_character_associations.tolist(),
                "character_cluster_labels": character_ufds.get_labels_for_connected_components(),
                "dialog_confidences": batch_dialog_confidence_scores[batch_index].tolist(),
            })
        return results
    
    def postprocess_ocr_tokens(self, generated_ids, skip_special_tokens=True):
        return self.ocr_preprocessor.batch_decode(generated_ids, skip_special_tokens=skip_special_tokens)
    
    # Обрезка изображения по заданным ограничивающим рамкам, корректируя их при необходимости, чтобы обеспечить корректный размер и положение обрезанных областей
    def crop_image(self, image, bboxes):
        crops_for_image = []
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox

            # fix the bounding box in case it is out of bounds or too small
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            x1, y1, x2, y2 = min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2) # just incase
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
            crops_for_image.append(crop)
        return crops_for_image

    # Список для хранения индексов персонажей
    def _get_indices_of_characters_to_keep(self, batch_scores, batch_labels, batch_bboxes, character_detection_threshold):
        indices_of_characters_to_keep = []
        for scores, labels, _ in zip(batch_scores, batch_labels, batch_bboxes):
            indices = torch.where((labels == 0) & (scores > character_detection_threshold))[0]
            indices_of_characters_to_keep.append(indices)
        return indices_of_characters_to_keep
    
    # Список для хранения индексов панелей
    def _get_indices_of_panels_to_keep(self, batch_scores, batch_labels, batch_bboxes, panel_detection_threshold):
        indices_of_panels_to_keep = []

        for scores, labels, bboxes in zip(batch_scores, batch_labels, batch_bboxes):
            indices = torch.where(labels == 2)[0]
            bboxes = bboxes[indices]
            scores = scores[indices]
            labels = labels[indices]
            if len(indices) == 0:
                indices_of_panels_to_keep.append([])
                continue
            scores, labels, indices, bboxes  = zip(*sorted(zip(scores, labels, indices, bboxes), reverse=True))
            panels_to_keep = []
            union_of_panels_so_far = box(0, 0, 0, 0)

            for ps, pb, pl, pi in zip(scores, bboxes, labels, indices):
                panel_polygon = box(pb[0], pb[1], pb[2], pb[3])
                # Пропуск панели с оценкой ниже заданного порога
                if ps < panel_detection_threshold:
                    continue
                # Пропуск панели, если их пересечение с уже найденными панелями больше 50% их площади
                if union_of_panels_so_far.intersection(panel_polygon).area / panel_polygon.area > 0.5:
                    continue
                panels_to_keep.append((ps, pl, pb, pi))
                union_of_panels_so_far = union_of_panels_so_far.union(panel_polygon)
            indices_of_panels_to_keep.append([p[3].item() for p in panels_to_keep])
        return indices_of_panels_to_keep
    
    # Cписок для хранения индексов текстовых блоков
    def _get_indices_of_texts_to_keep(self, batch_scores, batch_labels, batch_bboxes, text_detection_threshold):
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
            scores, labels, indices, bboxes  = zip(*sorted(zip(scores, labels, indices, bboxes), reverse=True))
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
                coco_annotation["annotations"].append({
                    "bbox": x1y1x2y2_to_xywh(bbox),
                    "category_id": label,
                    "area": (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]), # Площадь ограничивающей рамки
                })
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
            raise ValueError(
                f"{error_msg} Expected a List/Tuple, found {type(annotations)}."
            )
        if len(annotations) == 0:
            return
        if not isinstance(annotations[0], dict):
            raise ValueError(
                f"{error_msg} Expected a List[Dict], found {type(annotations[0])}."
            )
        if "image_id" not in annotations[0]:
            raise ValueError(
                f"{error_msg} Dict must contain 'image_id'."
            )
        if "bboxes_as_x1y1x2y2" not in annotations[0]:
            raise ValueError(
                f"{error_msg} Dict must contain 'bboxes_as_x1y1x2y2'."
            )
        if "labels" not in annotations[0]:
            raise ValueError(
                f"{error_msg} Dict must contain 'labels'."
            )
