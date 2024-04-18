from transformers import PreTrainedModel, VisionEncoderDecoderModel, ViTMAEModel, ConditionalDetrModel
from transformers.models.conditional_detr.modeling_conditional_detr import (
    ConditionalDetrMLPPredictionHead, 
    ConditionalDetrModelOutput,
    ConditionalDetrHungarianMatcher,
    inverse_sigmoid,
)
from .configuration_magi import MagiConfig
from .processing_magi import MagiProcessor
from torch import nn
from typing import Optional, List
import torch
from einops import rearrange, repeat, einsum
from .utils import move_to_device, visualise_single_image_prediction, sort_panels, sort_text_boxes_in_reading_order

class MagiModel(PreTrainedModel):
    config_class = MagiConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.processor = MagiProcessor(config)
        if not config.disable_ocr:
            self.ocr_model = VisionEncoderDecoderModel(config.ocr_model_config)
        if not config.disable_crop_embeddings:
            self.crop_embedding_model = ViTMAEModel(config.crop_embedding_model_config)
        if not config.disable_detections:
            self.num_non_obj_tokens = 5

            # Инициализация модели ConditionalDetrModel для детекции
            self.detection_transformer = ConditionalDetrModel(config.detection_model_config)

            # Инициализация заголовка предсказания для предсказания ограничивающих рамок
            self.bbox_predictor = ConditionalDetrMLPPredictionHead(
                input_dim=config.detection_model_config.d_model,
                hidden_dim=config.detection_model_config.d_model,
                output_dim=4, num_layers=3
            )
            ''' # Инициализация заголовка предсказания для определения, является ли текст диалогом
            self.is_this_text_a_dialogue = ConditionalDetrMLPPredictionHead(
                input_dim=config.detection_model_config.d_model,
                hidden_dim=config.detection_model_config.d_model,
                output_dim=1,
                num_layers=3
            )
            '''
            ''' # Инициализация заголовка предсказания для сопоставления персонажей и текста
            self.character_character_matching_head = ConditionalDetrMLPPredictionHead(
                input_dim = 3 * config.detection_model_config.d_model + (2 * config.crop_embedding_model_config.hidden_size if not config.disable_crop_embeddings else 0),
                hidden_dim=config.detection_model_config.d_model,
                output_dim=1, num_layers=3
            )
            '''
            ''' # Инициализация заголовка предсказания для сопоставления текста и персонажей
           self.text_character_matching_head = ConditionalDetrMLPPredictionHead(
                input_dim = 3 * config.detection_model_config.d_model,
                hidden_dim=config.detection_model_config.d_model,
                output_dim=1, num_layers=3
            )
            '''
        
            # Инициализация линейного классификатора для классификации классов
            self.class_labels_classifier = nn.Linear(
                config.detection_model_config.d_model, config.detection_model_config.num_labels
            )
            # Инициализация механизма сопоставления для детекции
            self.matcher = ConditionalDetrHungarianMatcher(
                class_cost=config.detection_model_config.class_cost,
                bbox_cost=config.detection_model_config.bbox_cost,
                giou_cost=config.detection_model_config.giou_cost
            )

    def move_to_device(self, input):
        return move_to_device(input, self.device)
    
    # Предсказание детектирования объектов на изображениях с использованием предварительно обученной модели
    def predict_detections_and_associations(
            self,
            images,
            move_to_device_fn=None,
            character_detection_threshold=0.3,
            panel_detection_threshold=0.2,
            text_detection_threshold=0.25,
            character_character_matching_threshold=0.65,
            text_character_matching_threshold=0.4,
        ):
        assert not self.config.disable_detections
        move_to_device_fn = self.move_to_device if move_to_device_fn is None else move_to_device_fn
        
        # Предварительная обработка входных изображений для детекции
        inputs_to_detection_transformer = self.processor.preprocess_inputs_for_detection(images)
        # Перемещение входных данных на устройство
        inputs_to_detection_transformer = move_to_device_fn(inputs_to_detection_transformer)
        
        # Получение выходных данных от трансформера детекции
        detection_transformer_output = self._get_detection_transformer_output(**inputs_to_detection_transformer)
        # Получение предсказанных ограничивающих рамок и классов
        predicted_class_scores, predicted_bboxes = self._get_predicted_bboxes_and_classes(detection_transformer_output)

        # Создание функции обратного вызова для получения оценок совпадения персонажей
        def get_character_character_matching_scores(batch_character_indices, batch_bboxes):
            # Получение предсказанных объектных токенов для каждого изображения в пакете
            predicted_obj_tokens_for_batch = self._get_predicted_obj_tokens(detection_transformer_output)
            # Получение предсказанных токенов совпадения персонажей (c2c) для каждого изображения в пакете
            predicted_c2c_tokens_for_batch = self._get_predicted_c2c_tokens(detection_transformer_output)
            # Извлечение ограничивающих рамок для персонажей из каждого изображения в пакете
            crop_bboxes = [batch_bboxes[i][batch_character_indices[i]] for i in range(len(batch_character_indices))]
            # Предсказание встраиваемых в персонажей эмбеддингов для каждого изображения в пакете
            crop_embeddings_for_batch = self.predict_crop_embeddings(images, crop_bboxes, move_to_device_fn)
            # Инициализация списков для хранения токенов объектов и токенов совпадения персонажей для каждого изображения в пакете
            character_obj_tokens_for_batch = []
            c2c_tokens_for_batch = []
            # Для каждого изображения в пакете
            for predicted_obj_tokens, predicted_c2c_tokens, character_indices in zip(predicted_obj_tokens_for_batch, predicted_c2c_tokens_for_batch, batch_character_indices):
                # Добавление токенов объектов персонажей в список
                character_obj_tokens_for_batch.append(predicted_obj_tokens[character_indices])
                # Добавление токенов совпадения персонажей в список
                c2c_tokens_for_batch.append(predicted_c2c_tokens)
            # Возвращение матриц аффинности совпадения персонажей
            return self._get_character_character_affinity_matrices(
                character_obj_tokens_for_batch=character_obj_tokens_for_batch,
                crop_embeddings_for_batch=crop_embeddings_for_batch,
                c2c_tokens_for_batch=c2c_tokens_for_batch,
                apply_sigmoid=True,
            )

    # Предсказания встраиваемых эмбеддингов для обрезанных изображений
    def predict_crop_embeddings(self, images, crop_bboxes, move_to_device_fn=None, mask_ratio=0.0, batch_size=256):
        # Если обрезка встраиваемых изображений отключена, возвращаем None
        if self.config.disable_crop_embeddings:
            return None
        
        # Проверка, что crop_bboxes является списком
        assert isinstance(crop_bboxes, List), "please provide a list of bboxes for each image to get embeddings for"
        
        # Если функция перемещения на устройство не предоставлена, используем метод класса
        move_to_device_fn = self.move_to_device if move_to_device_fn is None else move_to_device_fn
        
        # Временное изменение маскировочного коэффициента на указанный
        old_mask_ratio = self.crop_embedding_model.embeddings.config.mask_ratio
        self.crop_embedding_model.embeddings.config.mask_ratio = mask_ratio

        crops_per_image = []
        num_crops_per_batch = [len(bboxes) for bboxes in crop_bboxes]
        # Для каждого изображения и соответствующих ограничивающих рамок
        for image, bboxes, num_crops in zip(images, crop_bboxes, num_crops_per_batch):
            # Обрезка изображения по ограничивающим рамкам
            crops = self.processor.crop_image(image, bboxes)
            assert len(crops) == num_crops
            # Добавление обрезанных изображений в общий список
            crops_per_image.extend(crops)
        
        # Если обрезанных изображений нет, возвращаем пустые списки для каждого изображения
        if len(crops_per_image) == 0:
            return [[] for _ in crop_bboxes]

        # Предварительная обработка обрезанных изображений для получения встраиваемых эмбеддингов
        crops_per_image = self.processor.preprocess_inputs_for_crop_embeddings(crops_per_image)
        # Перемещение обрезанных изображений на устройство
        crops_per_image = move_to_device_fn(crops_per_image)
        
        # Обработка обрезанных изображений пакетами, чтобы избежать переполнения памяти
        embeddings = []
        for i in range(0, len(crops_per_image), batch_size):
            crops = crops_per_image[i:i+batch_size]
            # Получение встраиваемых эмбеддингов для каждого пакета
            embeddings_per_batch = self.crop_embedding_model(crops).last_hidden_state[:, 0]
            embeddings.append(embeddings_per_batch)
        embeddings = torch.cat(embeddings, dim=0)

        crop_embeddings_for_batch = []
        # Распределение полученных эмбеддингов по изображениям
        for num_crops in num_crops_per_batch:
            crop_embeddings_for_batch.append(embeddings[:num_crops])
            embeddings = embeddings[num_crops:]
        
        # Восстановление маскировочного коэффициента на значение по умолчанию
        self.crop_embedding_model.embeddings.config.mask_ratio = old_mask_ratio

        return crop_embeddings_for_batch


    def visualise_single_image_prediction(
                self, image_as_np_array, predictions, filename=None
        ):
            # Вызов функции визуализации предсказаний для одного изображения
            return visualise_single_image_prediction(image_as_np_array, predictions, filename)
    
    # Вычисления матриц аффинности совпадения текста и персонажей, а также матриц аффинности совпадения персонажей, используя предсказания модели и аннотации
    def get_affinity_matrices_given_annotations(
                self, images, annotations, move_to_device_fn=None, apply_sigmoid=True
        ):
        # Убедимся, что детекции не отключены
        assert not self.config.disable_detections
        # Если функция перемещения на устройство не предоставлена, используем метод класса
        move_to_device_fn = self.move_to_device if move_to_device_fn is None else move_to_device_fn

        # Извлечение ограничивающих рамок персонажей из аннотаций
        character_bboxes_in_batch = [[bbox for bbox, label in zip(a["bboxes_as_x1y1x2y2"], a["labels"]) if label == 0] for a in annotations]
        # Предсказание встраиваемых эмбеддингов для персонажей
        crop_embeddings_for_batch = self.predict_crop_embeddings(images, character_bboxes_in_batch, move_to_device_fn)

        # Предварительная обработка входных изображений и аннотаций для детекции
        inputs_to_detection_transformer = self.processor.preprocess_inputs_for_detection(images, annotations)
        inputs_to_detection_transformer = move_to_device_fn(inputs_to_detection_transformer)
        # Извлечение меток классов из предварительно обработанных входных данных
        processed_targets = inputs_to_detection_transformer.pop("labels")

        # Получение выходных данных от трансформера детекции
        detection_transformer_output = self._get_detection_transformer_output(**inputs_to_detection_transformer)
        # Получение предсказанных токенов объектов, токенов совпадения текста и персонажей
        predicted_obj_tokens_for_batch = self._get_predicted_obj_tokens(detection_transformer_output)
        #predicted_t2c_tokens_for_batch = self._get_predicted_t2c_tokens(detection_transformer_output)
        predicted_c2c_tokens_for_batch = self._get_predicted_c2c_tokens(detection_transformer_output)

        # Получение предсказанных ограничивающих рамок и классов
        predicted_class_scores, predicted_bboxes = self._get_predicted_bboxes_and_classes(detection_transformer_output)
        # Создание словаря для сопоставления
        matching_dict = {
            "logits": predicted_class_scores,
            "pred_boxes": predicted_bboxes,
        }
        # Сопоставление предсказаний с целевыми метками
        indices = self.matcher(matching_dict, processed_targets)

        # Инициализация списков для хранения соответствующих токенов объектов и эмбеддингов
        matched_char_obj_tokens_for_batch = []
        #matched_text_obj_tokens_for_batch = []
        #t2c_tokens_for_batch = []
        c2c_tokens_for_batch = []

        # Инициализация списков для хранения ограничивающих рамок текста и персонажей
        #text_bboxes_for_batch = []
        character_bboxes_for_batch = []

        # Для каждого изображения в пакете
        for j, (pred_idx, tgt_idx) in enumerate(indices):
            # Создание словаря для сопоставления индексов целевых и предсказанных объектов
            target_idx_to_pred_idx = {tgt.item(): pred.item() for pred, tgt in zip(pred_idx, tgt_idx)}
            targets_for_this_image = processed_targets[j]
            # Извлечение индексов текстовых и персонажеских ограничивающих рамок
            #indices_of_text_boxes_in_annotation = [i for i, label in enumerate(targets_for_this_image["class_labels"]) if label == 1]
            indices_of_char_boxes_in_annotation = [i for i, label in enumerate(targets_for_this_image["class_labels"]) if label == 0]
            # Получение индексов предсказанных текстовых и персонажеских объектов
            #predicted_text_indices = [target_idx_to_pred_idx[i] for i in indices_of_text_boxes_in_annotation]
            predicted_char_indices = [target_idx_to_pred_idx[i] for i in indices_of_char_boxes_in_annotation]
            
            # Добавление ограничивающих рамок текста и персонажей в списки
            '''text_bboxes_for_batch.append(
                [annotations[j]["bboxes_as_x1y1x2y2"][k] for k in indices_of_text_boxes_in_annotation]
            )'''
            character_bboxes_for_batch.append(
                [annotations[j]["bboxes_as_x1y1x2y2"][k] for k in indices_of_char_boxes_in_annotation]
            )
            
            # Добавление соответствующих токенов объектов в списки
            matched_char_obj_tokens_for_batch.append(predicted_obj_tokens_for_batch[j][predicted_char_indices])
            #matched_text_obj_tokens_for_batch.append(predicted_obj_tokens_for_batch[j][predicted_text_indices])
            #t2c_tokens_for_batch.append(predicted_t2c_tokens_for_batch[j])
            c2c_tokens_for_batch.append(predicted_c2c_tokens_for_batch[j])
        
        # Вычисление матриц аффинности совпадения текста и персонажей
        text_character_affinity_matrices = self._get_text_character_affinity_matrices(
            character_obj_tokens_for_batch=matched_char_obj_tokens_for_batch,
            #text_obj_tokens_for_this_batch=matched_text_obj_tokens_for_batch,
            #t2c_tokens_for_batch=t2c_tokens_for_batch,
            apply_sigmoid=apply_sigmoid,
        )

        # Вычисление матриц аффинности совпадения персонажей
        character_character_affinity_matrices = self._get_character_character_affinity_matrices(
            character_obj_tokens_for_batch=matched_char_obj_tokens_for_batch,
            crop_embeddings_for_batch=crop_embeddings_for_batch,
            c2c_tokens_for_batch=c2c_tokens_for_batch,
            apply_sigmoid=apply_sigmoid,
        )

        # Возвращение результатов
        return {
            "text_character_affinity_matrices": text_character_affinity_matrices,
            "character_character_affinity_matrices": character_character_affinity_matrices,
            #"text_bboxes_for_batch": text_bboxes_for_batch,
            "character_bboxes_for_batch": character_bboxes_for_batch,
        }   