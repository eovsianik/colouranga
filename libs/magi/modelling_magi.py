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
from numpy.typing import NDArray
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
            # Инициализация заголовка предсказания для определения, является ли текст диалогом
            self.is_this_text_a_dialogue = ConditionalDetrMLPPredictionHead(
                input_dim=config.detection_model_config.d_model,
                hidden_dim=config.detection_model_config.d_model,
                output_dim=1,
                num_layers=3
            )
            # Инициализация заголовка предсказания для сопоставления персонажей и текста
            self.character_character_matching_head = ConditionalDetrMLPPredictionHead(
                input_dim = 3 * config.detection_model_config.d_model + (2 * config.crop_embedding_model_config.hidden_size if not config.disable_crop_embeddings else 0),
                hidden_dim=config.detection_model_config.d_model,
                output_dim=1, num_layers=3
            )
            # Инициализация заголовка предсказания для сопоставления текста и персонажей
            self.text_character_matching_head = ConditionalDetrMLPPredictionHead(
                input_dim = 3 * config.detection_model_config.d_model,
                hidden_dim=config.detection_model_config.d_model,
                output_dim=1, num_layers=3
            )
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
            images: list[NDArray],
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

        # Создание функции обратного вызова для получения оценок совпадения текста и персонажей
        def get_text_character_matching_scores(batch_text_indices, batch_character_indices):
            # Получение предсказанных объектных токенов для каждого изображения в пакете
            predicted_obj_tokens_for_batch = self._get_predicted_obj_tokens(detection_transformer_output)
            # Получение предсказанных токенов совпадения текста и персонажей (t2c) для каждого изображения в пакете
            predicted_t2c_tokens_for_batch = self._get_predicted_t2c_tokens(detection_transformer_output)
            # Инициализация списков для хранения токенов объектов текста и персонажей для каждого изображения в пакете
            text_obj_tokens_for_batch = []
            character_obj_tokens_for_batch = []
            t2c_tokens_for_batch = []
            # Для каждого изображения в пакете
            for predicted_obj_tokens, predicted_t2c_tokens, text_indices, character_indices in zip(predicted_obj_tokens_for_batch, predicted_t2c_tokens_for_batch, batch_text_indices, batch_character_indices):
                # Добавление токенов объектов текста в список
                text_obj_tokens_for_batch.append(predicted_obj_tokens[text_indices])
                # Добавление токенов объектов персонажей в список
                character_obj_tokens_for_batch.append(predicted_obj_tokens[character_indices])
                # Добавление токенов совпадения текста и персонажей в список
                t2c_tokens_for_batch.append(predicted_t2c_tokens)
            # Возвращение матриц аффинности совпадения текста и персонажей
            return self._get_text_character_affinity_matrices(
                character_obj_tokens_for_batch=character_obj_tokens_for_batch,
                text_obj_tokens_for_this_batch=text_obj_tokens_for_batch,
                t2c_tokens_for_batch=t2c_tokens_for_batch,
                apply_sigmoid=True,
            )


        # Создание функции обратного вызова для получения оценок уверенности в диалоге
        def get_dialog_confidence_scores(batch_text_indices):
            # Получение предсказанных объектных токенов для каждого изображения в пакете
            predicted_obj_tokens_for_batch = self._get_predicted_obj_tokens(detection_transformer_output)
            # Инициализация списка для хранения оценок уверенности в диалоге для каждого изображения в пакете
            dialog_confidence = []
            # Для каждого изображения в пакете
            for predicted_obj_tokens, text_indices in zip(predicted_obj_tokens_for_batch, batch_text_indices):
                # Вычисление оценки уверенности в диалоге для каждого текстового объекта
                confidence = self.is_this_text_a_dialogue(predicted_obj_tokens[text_indices]).sigmoid()
                # Переупорядочивание оценок уверенности для соответствия формату
                dialog_confidence.append(rearrange(confidence, "i 1 -> i"))
            # Возвращение списка оценок уверенности в диалоге
            return dialog_confidence
  
        return self.processor.postprocess_detections_and_associations(
            predicted_bboxes=predicted_bboxes,
            predicted_class_scores=predicted_class_scores,
            original_image_sizes=torch.stack([torch.tensor(img.shape[:2]) for img in images], dim=0).to(predicted_bboxes.device),
            get_character_character_matching_scores=get_character_character_matching_scores,
            get_text_character_matching_scores=get_text_character_matching_scores,
            get_dialog_confidence_scores=get_dialog_confidence_scores,
            character_detection_threshold=character_detection_threshold,
            panel_detection_threshold=panel_detection_threshold,
            text_detection_threshold=text_detection_threshold,
            character_character_matching_threshold=character_character_matching_threshold,
            text_character_matching_threshold=text_character_matching_threshold,
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

    # Предсказания текста с помощью OCR для обрезанных изображений
    def predict_ocr(self, images, crop_bboxes, move_to_device_fn=None, use_tqdm=False, batch_size=32):
        # Убедимся, что OCR не отключен
        assert not self.config.disable_ocr
        # Если функция перемещения на устройство не предоставлена, используем метод класса
        move_to_device_fn = self.move_to_device if move_to_device_fn is None else move_to_device_fn

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

        # Предварительная обработка обрезанных изображений для OCR
        crops_per_image = self.processor.preprocess_inputs_for_ocr(crops_per_image)
        # Перемещение обрезанных изображений на устройство
        crops_per_image = move_to_device_fn(crops_per_image)
        
        # Обработка обрезанных изображений пакетами, чтобы избежать переполнения памяти
        all_generated_texts = []
        # Использование tqdm для отображения прогресса, если включено
        if use_tqdm:
            from tqdm import tqdm
            pbar = tqdm(range(0, len(crops_per_image), batch_size))
        else:
            pbar = range(0, len(crops_per_image), batch_size)
        for i in pbar:
            crops = crops_per_image[i:i+batch_size]
            # Генерация текста с помощью модели OCR
            generated_ids = self.ocr_model.generate(crops)
            # Постобработка сгенерированных токенов OCR
            generated_texts = self.processor.postprocess_ocr_tokens(generated_ids)
            all_generated_texts.extend(generated_texts)

        texts_for_images = []
        # Распределение сгенерированного текста по изображениям
        for num_crops in num_crops_per_batch:
            texts_for_images.append([x.replace("\n", "") for x in all_generated_texts[:num_crops]])
            all_generated_texts = all_generated_texts[num_crops:]

        return texts_for_images


    def visualise_single_image_prediction(
                self, image_as_np_array, predictions, filename=None
        ):
            # Вызов функции визуализации предсказаний для одного изображения
            return visualise_single_image_prediction(image_as_np_array, predictions, filename)

    def generate_transcript_for_single_image(
                self, predictions, ocr_results, filename=None
        ):
        # Извлечение меток кластеров персонажей и ассоциаций текста и персонажей из предсказаний
        character_clusters = predictions["character_cluster_labels"]
        text_to_character = predictions["text_character_associations"]
        # Преобразование ассоциаций текста и персонажей в словарь для удобства доступа
        text_to_character = {k: v for k, v in text_to_character}
        # Инициализация транскрипта
        transript = " ### Transcript ###\n"
        # Для каждого текста в результатах OCR
        for index, text in enumerate(ocr_results):
            # Если текст ассоциирован с персонажем
            if index in text_to_character:
                # Получение имени персонажа
                speaker = character_clusters[text_to_character[index]]
                # Форматирование имени персонажа
                speaker = f"<{speaker}>"
            else:
                # Если персонаж неизвестен
                speaker = "<?>"
            # Добавление строки с текстом и именем персонажа в транскрипт
            transript += f"{speaker}: {text}\n"
        # Если указан имя файла, сохранение транскрипта в файл
        if filename is not None:
            with open(filename, "w") as file:
                file.write(transript)
        # Возвращение сгенерированного транскрипта
        return transript

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
        predicted_t2c_tokens_for_batch = self._get_predicted_t2c_tokens(detection_transformer_output)
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
        matched_text_obj_tokens_for_batch = []
        t2c_tokens_for_batch = []
        c2c_tokens_for_batch = []

        # Инициализация списков для хранения ограничивающих рамок текста и персонажей
        text_bboxes_for_batch = []
        character_bboxes_for_batch = []

        # Для каждого изображения в пакете
        for j, (pred_idx, tgt_idx) in enumerate(indices):
            # Создание словаря для сопоставления индексов целевых и предсказанных объектов
            target_idx_to_pred_idx = {tgt.item(): pred.item() for pred, tgt in zip(pred_idx, tgt_idx)}
            targets_for_this_image = processed_targets[j]
            # Извлечение индексов текстовых и персонажеских ограничивающих рамок
            indices_of_text_boxes_in_annotation = [i for i, label in enumerate(targets_for_this_image["class_labels"]) if label == 1]
            indices_of_char_boxes_in_annotation = [i for i, label in enumerate(targets_for_this_image["class_labels"]) if label == 0]
            # Получение индексов предсказанных текстовых и персонажеских объектов
            predicted_text_indices = [target_idx_to_pred_idx[i] for i in indices_of_text_boxes_in_annotation]
            predicted_char_indices = [target_idx_to_pred_idx[i] for i in indices_of_char_boxes_in_annotation]
            
            # Добавление ограничивающих рамок текста и персонажей в списки
            text_bboxes_for_batch.append(
                [annotations[j]["bboxes_as_x1y1x2y2"][k] for k in indices_of_text_boxes_in_annotation]
            )
            character_bboxes_for_batch.append(
                [annotations[j]["bboxes_as_x1y1x2y2"][k] for k in indices_of_char_boxes_in_annotation]
            )
            
            # Добавление соответствующих токенов объектов в списки
            matched_char_obj_tokens_for_batch.append(predicted_obj_tokens_for_batch[j][predicted_char_indices])
            matched_text_obj_tokens_for_batch.append(predicted_obj_tokens_for_batch[j][predicted_text_indices])
            t2c_tokens_for_batch.append(predicted_t2c_tokens_for_batch[j])
            c2c_tokens_for_batch.append(predicted_c2c_tokens_for_batch[j])
        
        # Вычисление матриц аффинности совпадения текста и персонажей
        text_character_affinity_matrices = self._get_text_character_affinity_matrices(
            character_obj_tokens_for_batch=matched_char_obj_tokens_for_batch,
            text_obj_tokens_for_this_batch=matched_text_obj_tokens_for_batch,
            t2c_tokens_for_batch=t2c_tokens_for_batch,
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
            "text_bboxes_for_batch": text_bboxes_for_batch,
            "character_bboxes_for_batch": character_bboxes_for_batch,
        }

    # Получение встраиваний объектов, соответствующих заданным аннотациям, изображений
    def get_obj_embeddings_corresponding_to_given_annotations(
                self, images, annotations, move_to_device_fn=None
        ):
            # Проверяем, что обнаружение объектов не отключено в конфигурации
            assert not self.config.disable_detections
            # Определяем функцию для перемещения данных на устройство, если не указана
            move_to_device_fn = self.move_to_device if move_to_device_fn is None else move_to_device_fn

            # Предварительная обработка входных изображений и аннотаций для обнаружения объектов
            inputs_to_detection_transformer = self.processor.preprocess_inputs_for_detection(images, annotations)
            # Перемещаем данные на устройство, если указана функция
            inputs_to_detection_transformer = move_to_device_fn(inputs_to_detection_transformer)
            # Извлекаем метки из предварительно обработанных данных
            processed_targets = inputs_to_detection_transformer.pop("labels")

            # Получаем выходные данные трансформера для обнаружения объектов
            detection_transformer_output = self._get_detection_transformer_output(**inputs_to_detection_transformer)
            # Получаем предсказанные токены объектов, текста и панелей для каждого изображения в пакете
            predicted_obj_tokens_for_batch = self._get_predicted_obj_tokens(detection_transformer_output)
            predicted_t2c_tokens_for_batch = self._get_predicted_t2c_tokens(detection_transformer_output)
            predicted_c2c_tokens_for_batch = self._get_predicted_c2c_tokens(detection_transformer_output)

            # Получаем предсказанные оценки классов и ограничивающие рамки для объектов
            predicted_class_scores, predicted_bboxes = self._get_predicted_bboxes_and_classes(detection_transformer_output)
            # Создаем словарь для сопоставления предсказаний с целевыми метками
            matching_dict = {
                "logits": predicted_class_scores,
                "pred_boxes": predicted_bboxes,
            }
            # Сопоставляем предсказания с целевыми метками
            indices = self.matcher(matching_dict, processed_targets)

            # Инициализируем списки для хранения сопоставленных токенов объектов, текста и панелей
            matched_char_obj_tokens_for_batch = []
            matched_text_obj_tokens_for_batch = []
            matched_panel_obj_tokens_for_batch = []
            t2c_tokens_for_batch = []
            c2c_tokens_for_batch = []

            # Проходим по индексам сопоставлений для каждого изображения в пакете
            for j, (pred_idx, tgt_idx) in enumerate(indices):
                # Создаем словарь для сопоставления целевых индексов с предсказанными
                target_idx_to_pred_idx = {tgt.item(): pred.item() for pred, tgt in zip(pred_idx, tgt_idx)}
                # Получаем целевые метки для текущего изображения
                targets_for_this_image = processed_targets[j]
                # Определяем индексы символьных, текстовых и панельных боксов в аннотации
                indices_of_char_boxes_in_annotation = [i for i, label in enumerate(targets_for_this_image["class_labels"]) if label == 0]
                indices_of_text_boxes_in_annotation = [i for i, label in enumerate(targets_for_this_image["class_labels"]) if label == 1]
                indices_of_panel_boxes_in_annotation = [i for i, label in enumerate(targets_for_this_image["class_labels"]) if label == 2]
                # Получаем индексы предсказанных символьных, текстовых и панельных токенов
                predicted_text_indices = [target_idx_to_pred_idx[i] for i in indices_of_text_boxes_in_annotation]
                predicted_char_indices = [target_idx_to_pred_idx[i] for i in indices_of_char_boxes_in_annotation]
                predicted_panel_indices = [target_idx_to_pred_idx[i] for i in indices_of_panel_boxes_in_annotation]

                # Добавляем сопоставленные токены объектов, текста и панелей в соответствующие списки
                matched_char_obj_tokens_for_batch.append(predicted_obj_tokens_for_batch[j][predicted_char_indices])
                matched_text_obj_tokens_for_batch.append(predicted_obj_tokens_for_batch[j][predicted_text_indices])
                matched_panel_obj_tokens_for_batch.append(predicted_obj_tokens_for_batch[j][predicted_panel_indices])
                t2c_tokens_for_batch.append(predicted_t2c_tokens_for_batch[j])
                c2c_tokens_for_batch.append(predicted_c2c_tokens_for_batch[j])

            # Возвращаем словарь с сопоставленными токенами объектов, текста и панелей, а также токенами текста-к-тексту и текста-к-панели
            return {
                "character": matched_char_obj_tokens_for_batch,
                "text": matched_text_obj_tokens_for_batch,
                "panel": matched_panel_obj_tokens_for_batch,
                "t2c": t2c_tokens_for_batch,
                "c2c": c2c_tokens_for_batch,
            }

    # Сортировки боксов панелей и текстовых боксов в порядке чтения
    def sort_panels_and_text_bboxes_in_reading_order(
        self,
        batch_panel_bboxes,
        batch_text_bboxes,
    ):
        # Инициализируем списки для хранения отсортированных индексов панелей и текстовых боксов
        batch_sorted_panel_indices = []
        batch_sorted_text_indices = []
        # Проходим по каждому пакету данных
        for batch_index in range(len(batch_text_bboxes)):
            # Извлекаем боксы панелей и текстовых боксов для текущего пакета
            panel_bboxes = batch_panel_bboxes[batch_index]
            text_bboxes = batch_text_bboxes[batch_index]
            # Сортируем боксы панелей
            sorted_panel_indices = sort_panels(panel_bboxes)
            # Получаем отсортированные боксы панелей
            sorted_panels = [panel_bboxes[i] for i in sorted_panel_indices]
            # Сортируем текстовые боксы в порядке чтения, учитывая отсортированные панели
            sorted_text_indices = sort_text_boxes_in_reading_order(text_bboxes, sorted_panels)
            # Добавляем отсортированные индексы панелей и текстовых боксов в соответствующие списки
            batch_sorted_panel_indices.append(sorted_panel_indices)
            batch_sorted_text_indices.append(sorted_text_indices)
        # Возвращаем списки отсортированных индексов панелей и текстовых боксов
        return batch_sorted_panel_indices, batch_sorted_text_indices

    # Обнаружение объектов с использованием трансформера
    def _get_detection_transformer_output(
            self, 
            pixel_values: torch.FloatTensor,
            pixel_mask: Optional[torch.LongTensor] = None
    ):
        if self.config.disable_detections:
            raise ValueError("Detection model is disabled. Set disable_detections=False in the config.")
        return self.detection_transformer(
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            return_dict=True
        )
    
    # Извлекает предсказанные токены объектов из выходных данных трансформера
    def _get_predicted_obj_tokens(
                self,
                detection_transformer_output: ConditionalDetrModelOutput
        ):
            return detection_transformer_output.last_hidden_state[:, :-self.num_non_obj_tokens]

    # Извлекает предсказанные токены текста-к-тексту из выходных данных трансформера
    def _get_predicted_c2c_tokens(
                self,
                detection_transformer_output: ConditionalDetrModelOutput
        ):
            return detection_transformer_output.last_hidden_state[:, -self.num_non_obj_tokens]

    # Извлекает предсказанные токены текста-к-панели из выходных данных трансформера
    def _get_predicted_t2c_tokens(
                self,
                detection_transformer_output: ConditionalDetrModelOutput
        ):
            return detection_transformer_output.last_hidden_state[:, -self.num_non_obj_tokens+1]

    
    # Получает предсказанные оценки классов и ограничивающие рамки для объектов из выходных данных трансформера
    def _get_predicted_bboxes_and_classes(
                self,
                detection_transformer_output: ConditionalDetrModelOutput,
        ):
            # Проверяем, что обнаружение объектов не отключено в конфигурации
            if self.config.disable_detections:
                raise ValueError("Detection model is disabled. Set disable_detections=False in the config.")

            # Извлекаем предсказанные токены объектов
            obj = self._get_predicted_obj_tokens(detection_transformer_output)

            # Получаем предсказанные оценки классов объектов
            predicted_class_scores = self.class_labels_classifier(obj)
            # Получаем ссылочные точки для предсказанных боксов
            reference = detection_transformer_output.reference_points[:-self.num_non_obj_tokens] 
            # Преобразуем ссылочные точки обратно в координаты боксов
            reference_before_sigmoid = inverse_sigmoid(reference).transpose(0, 1)
            # Получаем предсказанные ограничивающие рамки для объектов
            predicted_boxes = self.bbox_predictor(obj)
            # Сдвигаем предсказанные боксы относительно ссылочных точек
            predicted_boxes[..., :2] += reference_before_sigmoid
            # Применяем сигмоиду к координатам боксов
            predicted_boxes = predicted_boxes.sigmoid()

            # Возвращаем предсказанные оценки классов и ограничивающие рамки для объектов
            return predicted_class_scores, predicted_boxes

    
    # Вычисляет матрицы аффинности между символами, используя токены символов и токены текста-к-тексту
    def _get_character_character_affinity_matrices(
                self,
                character_obj_tokens_for_batch: List[torch.FloatTensor] = None,
                crop_embeddings_for_batch: List[torch.FloatTensor] = None,
                c2c_tokens_for_batch: List[torch.FloatTensor] = None,
                apply_sigmoid=True,
        ):
            # Проверяем, что обнаружение объектов или встраивания не отключено в конфигурации
            assert self.config.disable_detections or (character_obj_tokens_for_batch is not None and c2c_tokens_for_batch is not None)
            assert self.config.disable_crop_embeddings or crop_embeddings_for_batch is not None
            assert not self.config.disable_detections or not self.config.disable_crop_embeddings

            if self.config.disable_detections:
                # Если обнаружение объектов отключено, вычисляем матрицы аффинности на основе встраиваний
                affinity_matrices = []
                for crop_embeddings in crop_embeddings_for_batch:
                    # Нормализуем встраивания
                    crop_embeddings = crop_embeddings / crop_embeddings.norm(dim=-1, keepdim=True)
                    # Вычисляем матрицу аффинности между встраиваниями
                    affinity_matrix = einsum("i d, j d -> i j", crop_embeddings, crop_embeddings)
                    affinity_matrices.append(affinity_matrix)
                return affinity_matrices
            affinity_matrices = []
            for batch_index, (character_obj_tokens, c2c) in enumerate(zip(character_obj_tokens_for_batch, c2c_tokens_for_batch)):
                if character_obj_tokens.shape[0] == 0:
                    # Если нет токенов символов, добавляем пустую матрицу аффинности
                    affinity_matrices.append(torch.zeros(0, 0).type_as(character_obj_tokens))
                    continue
                if not self.config.disable_crop_embeddings:
                    # Если встраивания не отключены, добавляем их к токенам символов
                    crop_embeddings = crop_embeddings_for_batch[batch_index]
                    assert character_obj_tokens.shape[0] == crop_embeddings.shape[0]
                    character_obj_tokens = torch.cat([character_obj_tokens, crop_embeddings], dim=-1)
                # Создаем матрицы для каждой пары символов
                char_i = repeat(character_obj_tokens, "i d -> i repeat d", repeat=character_obj_tokens.shape[0])
                char_j = repeat(character_obj_tokens, "j d -> repeat j d", repeat=character_obj_tokens.shape[0])
                char_ij = rearrange([char_i, char_j], "two i j d -> (i j) (two d)")
                # Добавляем токены текста-к-тексту к матрице
                c2c = repeat(c2c, "d -> repeat d", repeat = char_ij.shape[0])
                char_ij_c2c = torch.cat([char_ij, c2c], dim=-1)
                # Вычисляем аффинности между символами
                character_character_affinities = self.character_character_matching_head(char_ij_c2c)
                # Перестраиваем аффинности в матрицу
                character_character_affinities = rearrange(character_character_affinities, "(i j) 1 -> i j", i=char_i.shape[0])
                # Сделаем матрицу симметричной
                character_character_affinities = (character_character_affinities + character_character_affinities.T) / 2
                if apply_sigmoid:
                    # Применяем сигмоиду к аффинностям
                    character_character_affinities = character_character_affinities.sigmoid()
                affinity_matrices.append(character_character_affinities)
            return affinity_matrices

   # Вычисляет матрицы аффинности между текстом и символами, используя токены символов, токены текста и токены текста-к-символу
    def _get_text_character_affinity_matrices(
                self,
                character_obj_tokens_for_batch: List[torch.FloatTensor] = None,
                text_obj_tokens_for_this_batch: List[torch.FloatTensor] = None,
                t2c_tokens_for_batch: List[torch.FloatTensor] = None,
                apply_sigmoid=True,
        ):
            # Проверяем, что обнаружение объектов не отключено в конфигурации
            assert not self.config.disable_detections
            # Проверяем, что предоставлены все необходимые данные
            assert character_obj_tokens_for_batch is not None and text_obj_tokens_for_this_batch is not None and t2c_tokens_for_batch is not None
            affinity_matrices = []
            for character_obj_tokens, text_obj_tokens, t2c in zip(character_obj_tokens_for_batch, text_obj_tokens_for_this_batch, t2c_tokens_for_batch):
                # Если нет токенов символов или текста, добавляем пустую матрицу аффинности
                if character_obj_tokens.shape[0] == 0 or text_obj_tokens.shape[0] == 0:
                    affinity_matrices.append(torch.zeros(text_obj_tokens.shape[0], character_obj_tokens.shape[0]).type_as(character_obj_tokens))
                    continue
                # Создаем матрицы для каждой пары текста и символа
                text_i = repeat(text_obj_tokens, "i d -> i repeat d", repeat=character_obj_tokens.shape[0])
                char_j = repeat(character_obj_tokens, "j d -> repeat j d", repeat=text_obj_tokens.shape[0])
                text_char = rearrange([text_i, char_j], "two i j d -> (i j) (two d)")
                # Добавляем токены текста-к-символу к матрице
                t2c = repeat(t2c, "d -> repeat d", repeat = text_char.shape[0])
                text_char_t2c = torch.cat([text_char, t2c], dim=-1)
                # Вычисляем аффинности между текстом и символами
                text_character_affinities = self.text_character_matching_head(text_char_t2c)
                # Перестраиваем аффинности в матрицу
                text_character_affinities = rearrange(text_character_affinities, "(i j) 1 -> i j", i=text_i.shape[0])
                if apply_sigmoid:
                    # Применяем сигмоиду к аффинностям
                    text_character_affinities = text_character_affinities.sigmoid()
                affinity_matrices.append(text_character_affinities)
            return affinity_matrices
