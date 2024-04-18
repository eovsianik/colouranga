from typing import List, Optional

import torch
from libs.colouranga.from_magi_model.config import MagiConfig
from libs.colouranga.from_magi_model.processor import MagiProcessor
from libs.colouranga.from_magi_model.utils import (
    move_to_device,
    sort_panels,
    sort_text_boxes_in_reading_order,
    visualise_single_image_prediction,
)
from numpy.typing import NDArray
from torch import nn
from transformers import (
    ConditionalDetrModel,
    PreTrainedModel,
    ViTMAEModel,
)
from transformers.models.conditional_detr.modeling_conditional_detr import (
    ConditionalDetrHungarianMatcher,
    ConditionalDetrMLPPredictionHead,
    ConditionalDetrModelOutput,
    inverse_sigmoid,
)


class MyMagiModel(PreTrainedModel):
    config_class = MagiConfig

    def __init__(self, config: MagiConfig):
        super().__init__(config)
        self.config = config
        self.processor = MagiProcessor(config)
        self.crop_embedding_model = ViTMAEModel(config.crop_embedding_model_config)
        self.num_non_obj_tokens = 5

        # Инициализация модели ConditionalDetrModel для детекции
        self.detection_transformer = ConditionalDetrModel(config.detection_model_config)  # type: ignore

        # Инициализация заголовка предсказания для предсказания ограничивающих рамок
        self.bbox_predictor = ConditionalDetrMLPPredictionHead(
            input_dim=config.detection_model_config.d_model,
            hidden_dim=config.detection_model_config.d_model,
            output_dim=4,
            num_layers=3,
        )
        # ? Инициализация заголовка предсказания для определения, является ли текст диалогом
        # self.is_this_text_a_dialogue = ConditionalDetrMLPPredictionHead(
        #     input_dim=config.detection_model_config.d_model,
        #     hidden_dim=config.detection_model_config.d_model,
        #     output_dim=1,
        #     num_layers=3,
        # )
        # Инициализация заголовка предсказания для сопоставления персонажей и текста
        self.character_character_matching_head = ConditionalDetrMLPPredictionHead(
            input_dim=3 * config.detection_model_config.d_model
            + (2 * config.crop_embedding_model_config.hidden_size),
            hidden_dim=config.detection_model_config.d_model,
            output_dim=1,
            num_layers=3,
        )
        # Инициализация линейного классификатора
        self.class_labels_classifier = nn.Linear(
            config.detection_model_config.d_model, config.detection_model_config.num_labels
        )
        # Инициализация механизма сопоставления для детекции
        self.matcher = ConditionalDetrHungarianMatcher(
            class_cost=config.detection_model_config.class_cost,
            bbox_cost=config.detection_model_config.bbox_cost,
            giou_cost=config.detection_model_config.giou_cost,
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

        move_to_device_fn = self.move_to_device if move_to_device_fn is None else move_to_device_fn

        # Предварительная обработка входных изображений для детекции
        inputs_to_detection_transformer = self.processor.preprocess_inputs_for_detection(images)
        # Перемещение входных данных на устройство
        inputs_to_detection_transformer = move_to_device_fn(inputs_to_detection_transformer) # dict, len 2

        # Получение выходных данных от трансформера детекции
        detection_transformer_output = self._get_detection_transformer_output(
            **inputs_to_detection_transformer  # type: ignore
        ) # detection_transformer_output - <class 'transformers.models.conditional_detr.modeling_conditional_detr.ConditionalDetrModelOutput'> len = 3
        # Получение предсказанных ограничивающих рамок и классов
        predicted_class_scores, predicted_bboxes = self._get_predicted_bboxes_and_classes(
            detection_transformer_output
        )
        # predicted_class_scores - torch.Tensor [кол-во картинок, 300, 3]
        # predicted_bboxes - torch.Tensor [кол-во картинок, 300, 4]

        # Создание функции обратного вызова для получения оценок совпадения персонажей
        def get_character_character_matching_scores(batch_character_indices, batch_bboxes):
            # Получение предсказанных объектных токенов для каждого изображения в пакете
            predicted_obj_tokens_for_batch = self._get_predicted_obj_tokens(
                detection_transformer_output
            )
            # predicted_obj_tokens_for_batch - [кол-во картинок, 300, 256]
            # detection_transformer_output - len 3

            # Получение предсказанных токенов совпадения персонажей (c2c) для каждого изображения в пакете
            predicted_c2c_tokens_for_batch = self._get_predicted_c2c_tokens(
                detection_transformer_output
            )
            # predicted_c2c_tokens_for_batch - [кол-во картинок, 256]
            
            # Извлечение ограничивающих рамок для персонажей из каждого изображения в пакете
            # По каждой картинке координаты получаем
            crop_bboxes = [
                batch_bboxes[i][batch_character_indices[i]]
                for i in range(len(batch_character_indices))
            ]


            # Предсказание встраиваемых в персонажей эмбеддингов для каждого изображения в пакете
            crop_embeddings_for_batch = self.predict_crop_embeddings(
                images, crop_bboxes, move_to_device_fn
            )
            # crop_embeddings_for_batch - список с тензорами
            # каждый тензор имеет размер: [кол-во распознанных bbox на изображении/странице, 768]
            # Инициализация списков для хранения токенов объектов и токенов совпадения персонажей для каждого изображения в пакете
            character_obj_tokens_for_batch = []
            c2c_tokens_for_batch = []
            # Для каждого изображения в пакете
            for predicted_obj_tokens, predicted_c2c_tokens, character_indices in zip(
                predicted_obj_tokens_for_batch,
                predicted_c2c_tokens_for_batch,
                batch_character_indices,
            ):
                # predicted_obj_tokens_for_batch - [кол-во изображений/страниц, 300, 256] - тут числа меньше 1 формата 10 в степени минус
                # predicted_c2c_tokens_for_batch - [кол-во изображений/страниц, 256] - тут числа меньше 1 формата float
                # batch_character_indices - list с тензорами, где каждый тензор содержит в себе индексы bbox с персами для каждого изображения
                # Добавление токенов объектов персонажей в список
                character_obj_tokens_for_batch.append(predicted_obj_tokens[character_indices])
                # character_obj_tokens_for_batch - list с тензорами
                # размер каждого тензора: [кол-во всех bbox на изображении/странице, 256]
                # Добавление токенов совпадения персонажей в список
                c2c_tokens_for_batch.append(predicted_c2c_tokens)
                # list длиной в количество страниц с тензорами [256]
            # Возвращение матриц аффинности совпадения персонажей
            return self._get_character_character_affinity_matrices(
                character_obj_tokens_for_batch=character_obj_tokens_for_batch,
                crop_embeddings_for_batch=crop_embeddings_for_batch,  # type: ignore
                c2c_tokens_for_batch=c2c_tokens_for_batch,
                apply_sigmoid=True,
            )

        return self.processor.postprocess_detections_and_associations(
            images=images,
            predicted_bboxes=predicted_bboxes,
            # predicted_bboxes [кол-во картинок, 300, 4]
            predicted_class_scores=predicted_class_scores,
            # predicted_class_scores [кол-во картинок, 300, 3]
            original_image_sizes=torch.stack(
                [torch.tensor(img.shape[:2]) for img in images], dim=0
            ).to(predicted_bboxes.device),
            # original_image_sizes [кол-во картинок, 2]
            get_character_character_matching_scores=get_character_character_matching_scores,
            character_detection_threshold=character_detection_threshold,
            panel_detection_threshold=panel_detection_threshold,
            text_detection_threshold=text_detection_threshold,
            character_character_matching_threshold=character_character_matching_threshold,
        )

    # Предсказания встраиваемых эмбеддингов для обрезанных изображений
    def predict_crop_embeddings(
        self, images, crop_bboxes, move_to_device_fn=None, mask_ratio=0.0, batch_size=256
    ):
        # Проверка, что crop_bboxes является списком
        assert isinstance(
            crop_bboxes, List
        ), "please provide a list of bboxes for each image to get embeddings for"

        # Если функция перемещения на устройство не предоставлена, используем метод класса
        move_to_device_fn = self.move_to_device if move_to_device_fn is None else move_to_device_fn

        # Временное изменение маскировочного коэффициента на указанный
        # заглушка, так как другое значение в трансформере
        old_mask_ratio = self.crop_embedding_model.embeddings.config.mask_ratio
        self.crop_embedding_model.embeddings.config.mask_ratio = mask_ratio

        crops_per_image = []
        # num_crops_per_batch - список с числами, каждое число - количество боксов с персонажами на одной картинке/странице
        num_crops_per_batch = [len(bboxes) for bboxes in crop_bboxes]
        # crop_bboxes - len кол-во картинок - несколько тензоров, где указаны координаты bbox
        # bboxes - это тензоры для каждой картинки, где есть координаты для каждого bbox - bboxes постоянно обновляются
        # num_crops_per_batch - список из цифр, где указано: сколько bbox с персами на каждой картинке
        # Для каждого изображения и соответствующих ограничивающих рамок
        for image, bboxes, num_crops in zip(images, crop_bboxes, num_crops_per_batch):
            # это для обработки каждой страницы по отдельности
            # Обрезка изображения по ограничивающим рамкам
            crops = self.processor.crop_image(image, bboxes)
            # это местный процессор - MagiProcessor - crop_image - внутри процессора
            # crops - список длиной в количество bbox с персами с numpy arrays
            # numpy arrays - все пиксели bbox (длина(?), ширина(?), 3 канала) 
            assert len(crops) == num_crops
            # Добавление обрезанных изображений в общий список
            crops_per_image.extend(crops)
            # все обработанные bbox добавляют в этот список
            # crops_per_image - список - list

        # Если обрезанных bbox нет, возвращаем пустые списки для каждого изображения
        if len(crops_per_image) == 0:
            return [[] for _ in crop_bboxes]

        # Предварительная обработка обрезанных изображений для получения встраиваемых эмбеддингов
        crops_per_image = self.processor.preprocess_inputs_for_crop_embeddings(crops_per_image)
        # ATTENTION!!! ВАЖНО!!! - тут список где абсолютно все bboxes, то есть они не разделяются по каждому отдельному изображению
        # они все вместе в списке в виде numpy arrays
        # Перемещение обрезанных изображений на устройство
        crops_per_image = move_to_device_fn(crops_per_image)

        # Обработка обрезанных bbox пакетами, чтобы избежать переполнения памяти
        embeddings = []
        for i in range(0, len(crops_per_image), batch_size):
            # crops_per_image - torchTensor [количество всех вообще bbox, 3, 224, 224]
            # batch_size = 256 = int - шаг в цикле
            crops = crops_per_image[i : i + batch_size]
            # crops - torchTensor [количество всех вообще bbox, 3, 224, 224]
            # мб итоговый размер иной, но у нас всего 5 картинок
            # Получение встраиваемых эмбеддингов для каждого пакета
            embeddings_per_batch = self.crop_embedding_model(crops).last_hidden_state[:, 0]
            # embeddings_per_batch - [количество всех вообще bbox, 768]
            embeddings.append(embeddings_per_batch)
        embeddings = torch.cat(embeddings, dim=0)

        crop_embeddings_for_batch = []
        # Распределение полученных эмбеддингов по изображениям
        for num_crops in num_crops_per_batch:
            crop_embeddings_for_batch.append(embeddings[:num_crops])
            # crop_embeddings_for_batch - list с тензорами
            embeddings = embeddings[num_crops:]
            # embeddings torchTensor [количество всех bbox, 768]

        # Восстановление маскировочного коэффициента на значение по умолчанию
        # тут они возвращают какую-то константу, которая по умолчанию стояла в трансформере
        self.crop_embedding_model.embeddings.config.mask_ratio = old_mask_ratio

        return crop_embeddings_for_batch

    def visualise_single_image_prediction(self, image_as_np_array, predictions, filename=None):
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
        character_bboxes_in_batch = [
            [bbox for bbox, label in zip(a["bboxes_as_x1y1x2y2"], a["labels"]) if label == 0]
            for a in annotations
        ]
        # Предсказание встраиваемых эмбеддингов для персонажей
        crop_embeddings_for_batch = self.predict_crop_embeddings(
            images, character_bboxes_in_batch, move_to_device_fn
        )

        # Предварительная обработка входных изображений и аннотаций для детекции
        inputs_to_detection_transformer = self.processor.preprocess_inputs_for_detection(
            images, annotations
        )
        inputs_to_detection_transformer = move_to_device_fn(inputs_to_detection_transformer)
        # Извлечение меток классов из предварительно обработанных входных данных
        processed_targets = inputs_to_detection_transformer.pop("labels")  # type: ignore

        # Получение выходных данных от трансформера детекции
        detection_transformer_output = self._get_detection_transformer_output(
            **inputs_to_detection_transformer  # type: ignore
        )
        # Получение предсказанных токенов объектов, токенов совпадения текста и персонажей
        predicted_obj_tokens_for_batch = self._get_predicted_obj_tokens(
            detection_transformer_output
        )
        predicted_c2c_tokens_for_batch = self._get_predicted_c2c_tokens(
            detection_transformer_output
        )

        # Получение предсказанных ограничивающих рамок и классов
        predicted_class_scores, predicted_bboxes = self._get_predicted_bboxes_and_classes(
            detection_transformer_output
        )
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
        c2c_tokens_for_batch = []

        # Инициализация списков для хранения ограничивающих рамок текста и персонажей
        text_bboxes_for_batch = []
        character_bboxes_for_batch = []

        # Для каждого изображения в пакете
        for j, (pred_idx, tgt_idx) in enumerate(indices):
            # Создание словаря для сопоставления индексов целевых и предсказанных объектов
            target_idx_to_pred_idx = {
                tgt.item(): pred.item() for pred, tgt in zip(pred_idx, tgt_idx)
            }
            targets_for_this_image = processed_targets[j]
            # Извлечение индексов текстовых и персонажеских ограничивающих рамок
            indices_of_text_boxes_in_annotation = [
                i for i, label in enumerate(targets_for_this_image["class_labels"]) if label == 1
            ]
            indices_of_char_boxes_in_annotation = [
                i for i, label in enumerate(targets_for_this_image["class_labels"]) if label == 0
            ]
            # Получение индексов предсказанных текстовых и персонажеских объектов
            predicted_text_indices = [
                target_idx_to_pred_idx[i] for i in indices_of_text_boxes_in_annotation
            ]
            predicted_char_indices = [
                target_idx_to_pred_idx[i] for i in indices_of_char_boxes_in_annotation
            ]

            # Добавление ограничивающих рамок текста и персонажей в списки
            text_bboxes_for_batch.append(
                [
                    annotations[j]["bboxes_as_x1y1x2y2"][k]
                    for k in indices_of_text_boxes_in_annotation
                ]
            )
            character_bboxes_for_batch.append(
                [
                    annotations[j]["bboxes_as_x1y1x2y2"][k]
                    for k in indices_of_char_boxes_in_annotation
                ]
            )

            # Добавление соответствующих токенов объектов в списки
            matched_char_obj_tokens_for_batch.append(
                predicted_obj_tokens_for_batch[j][predicted_char_indices]
            )
            matched_text_obj_tokens_for_batch.append(
                predicted_obj_tokens_for_batch[j][predicted_text_indices]
            )
            c2c_tokens_for_batch.append(predicted_c2c_tokens_for_batch[j])

        # Вычисление матриц аффинности совпадения персонажей
        character_character_affinity_matrices = self._get_character_character_affinity_matrices(
            character_obj_tokens_for_batch=matched_char_obj_tokens_for_batch,
            crop_embeddings_for_batch=crop_embeddings_for_batch,  # type: ignore
            c2c_tokens_for_batch=c2c_tokens_for_batch,
            apply_sigmoid=apply_sigmoid,
        )

        # Возвращение результатов
        return {
            "character_character_affinity_matrices": character_character_affinity_matrices,
            "text_bboxes_for_batch": text_bboxes_for_batch,
            "character_bboxes_for_batch": character_bboxes_for_batch,
        }

    # Обнаружение объектов с использованием трансформера - тут просто суют в трансформер, мб можно и заменить на сразу сунуть в тарнсфомрер
    def _get_detection_transformer_output(
        self, pixel_values: torch.FloatTensor, pixel_mask: Optional[torch.LongTensor] = None
    ):
        return self.detection_transformer(
            pixel_values=pixel_values, pixel_mask=pixel_mask, return_dict=True
        ) # pixel_values - torch.Tenson torch.Size([5, 3, 1149, 800])  pixel_mask - torch.Tenson torch.Size([5,1149, 800])

    # Извлекает предсказанные токены объектов из выходных данных трансформера мб не стоит создавать отдельную функцию и просто включить в код
    def _get_predicted_obj_tokens(self, detection_transformer_output: ConditionalDetrModelOutput):
        return detection_transformer_output.last_hidden_state[:, : -self.num_non_obj_tokens]
        # num_non_obj_tokens = количество изображений 
    # Извлекает предсказанные токены текста-к-тексту из выходных данных трансформера
    def _get_predicted_c2c_tokens(self, detection_transformer_output: ConditionalDetrModelOutput):
        return detection_transformer_output.last_hidden_state[:, -self.num_non_obj_tokens]
        # [кол-во картинок, 256]

    # Получает предсказанные оценки классов и ограничивающие рамки для объектов из выходных данных трансформера
    def _get_predicted_bboxes_and_classes(
        self,
        detection_transformer_output: ConditionalDetrModelOutput,
    ):
        # Извлекаем предсказанные токены объектов
        obj = self._get_predicted_obj_tokens(detection_transformer_output)
        # obj - torch.Tensor [кол-во картинок, 300, 256]
        # Получаем предсказанные оценки классов объектов
        predicted_class_scores = self.class_labels_classifier(obj)
        # predicted_class_scores - torch.Tensor [кол-во картинок, 300, 3]
        # сам class_labels_classifier - nn.Linear(params)
        # Получаем ссылочные точки для предсказанных боксов
        reference = detection_transformer_output.reference_points[: -self.num_non_obj_tokens]  # type: ignore
        # reference - torch.Tensor [300, кол-во картинок, 2]
        # Преобразуем ссылочные точки обратно в координаты боксов
        reference_before_sigmoid = inverse_sigmoid(reference).transpose(0, 1)
        # reference_before_sigmoid torch.Tensor [кол-во картинок, 300, 2]
        # inverse_sigmoid - написанная функция в DETR
        # Получаем предсказанные ограничивающие рамки для объектов
        predicted_boxes = self.bbox_predictor(obj)
        # predicted_boxes - torch.Tensor [кол-во картинок,300, 4]
        # Сдвигаем предсказанные боксы относительно ссылочных точек
        predicted_boxes[..., :2] += reference_before_sigmoid
        # Применяем сигмоиду к координатам боксов
        predicted_boxes = predicted_boxes.sigmoid()
        # predicted_boxes - torch.Tensor [кол-во картинок,300, 4]
        # Возвращаем предсказанные оценки классов и ограничивающие рамки для объектов
        return predicted_class_scores, predicted_boxes

    # Вычисляет матрицы аффинности между символами, используя токены символов и токены текста-к-тексту
    def _get_character_character_affinity_matrices(
        self,
        character_obj_tokens_for_batch: list[torch.FloatTensor] | None = None,
        crop_embeddings_for_batch: list[torch.FloatTensor] | None = None,
        c2c_tokens_for_batch: list[torch.FloatTensor] | None = None,
        apply_sigmoid=True,
    ):
        affinity_matrices = []
        for batch_index, (character_obj_tokens, c2c) in enumerate(
            zip(character_obj_tokens_for_batch, c2c_tokens_for_batch)  # type: ignore
        ):
            if character_obj_tokens.shape[0] == 0:
                # Если нет токенов символов, добавляем пустую матрицу аффинности
                affinity_matrices.append(torch.zeros(0, 0).type_as(character_obj_tokens))
                continue
            if crop_embeddings_for_batch is not None:
                # Если встраивания не отключены, добавляем их к токенам символов
                crop_embeddings = crop_embeddings_for_batch[batch_index]
                # crop_embeddings - тензор [кол-во bbox на странице, 768]
                assert character_obj_tokens.shape[0] == crop_embeddings.shape[0]
                character_obj_tokens = torch.cat([character_obj_tokens, crop_embeddings], dim=-1)
            # Создаем матрицы для каждой пары символов
            char_i = character_obj_tokens.unsqueeze(1)
            # char_i [кол-во bbox на странице, 1, 768]
            char_i = char_i.expand(char_i.size(dim=0), char_i.size(dim=0), 1024)
            # char_i [кол-во bbox на странице, кол-во bbox на странице, 768]
            char_j = character_obj_tokens.unsqueeze(0)
            # char_j [1, кол-во bbox на странице, 1024]
            char_j = char_j.expand(char_j.size(dim=1), char_j.size(dim=1), 1024)
            # char_i [кол-во bbox на странице, кол-во bbox на странице, 1024]
            char_ij = torch.cat((char_i, char_j), dim=-1)
            # char_ij [кол-во bbox на странице, кол-во bbox на странице, 2048]
            char_ij = char_ij.view(-1, 2048)
            # char_ij [кол-во bbox на странице^2, 2048]
            # Добавляем токены текста-к-тексту к матрице
            c2c = torch.tile(c2c, (char_ij.shape[0], 1))
            # c2c [кол-во bbox на странице^2, 256] - но вообще ранее было что-то непонятное
            char_ij_c2c = torch.cat([char_ij, c2c], dim=-1)
            # char_ij_c2c [кол-во bbox на странице^2, 2304]
            # Вычисляем аффинности между символами
            character_character_affinities = self.character_character_matching_head(char_ij_c2c)
            # [кол-во bbox на странице^2, 1]
            # Перестраиваем аффинности в матрицу
            character_character_affinities = character_character_affinities.view(char_i.shape[0], -1)
            # # [кол-во bbox на странице, кол-во bbox на странице]
            # Сделаем матрицу симметричной
            character_character_affinities = (character_character_affinities + character_character_affinities.T).div(2)
            if apply_sigmoid:
                # Применяем сигмоиду к аффинностям
                character_character_affinities = character_character_affinities.sigmoid()
            affinity_matrices.append(character_character_affinities)
        return affinity_matrices
