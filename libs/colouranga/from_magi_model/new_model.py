from re import I
from typing import List, Optional

import torch
from libs.colouranga.from_magi_model.config import MagiConfig
from libs.colouranga.from_magi_model.processor import MagiProcessor
from libs.colouranga.from_magi_model.utils import (
    move_to_device,
    visualise_single_image_prediction,
)
from numpy import indices
from numpy.typing import NDArray
from torch import nn
from transformers import (
    ConditionalDetrModel,
    PreTrainedModel,
    ViTMAEModel,
)
from transformers.image_transforms import center_to_corners_format
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

    def get_detr_output(
        self,
        images: list[NDArray],
    ):
        move_to_device_fn = self.move_to_device

        inputs_to_detection_transformer = self.processor.preprocess_inputs_for_detection(images)
        inputs_to_detection_transformer = move_to_device_fn(
            inputs_to_detection_transformer
        )  # dict, len 2

        detection_transformer_output = self._get_detection_transformer_output(
            **inputs_to_detection_transformer  # type: ignore
        )
        predicted_class_scores, predicted_bboxes = self._get_predicted_bboxes_and_classes(
            detection_transformer_output
        )

        return predicted_class_scores, predicted_bboxes

    def get_batch_bboxes_and_batch_character_indices(
        self,
        predicted_bboxes,
        predicted_class_scores,
        original_image_sizes,  # [кол-во картинок, ширина и длина]
        character_detection_threshold=0.3,
    ):
        # Получение максимальные оценки и метки для каждого предсказания
        batch_scores, batch_labels = predicted_class_scores.max(-1)
        # batch_scores [кол-во картинок, 300]
        # batch_labels [кол-во картинок, 300]

        batch_scores = batch_scores.sigmoid()
        batch_labels = batch_labels.long()

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

        batch_character_indices = self.processor._get_indices_of_characters_to_keep(
            batch_scores, batch_labels, batch_bboxes, character_detection_threshold
        )  # list - длина равна количеству картинок,элементы - тензоры с индексами

        return batch_bboxes, batch_character_indices, batch_scores

    def get_crops_and_embeddings(self, images: list[NDArray]):
        predicted_class_scores, predicted_bboxes = self.get_detr_output(images)

        # predicted_class_scores = _get_predicted_bboxes_and_classes_new(?)

        original_image_sizes = torch.stack(
            [torch.tensor(img.shape[:2]) for img in images], dim=0
        ).to(predicted_bboxes.device)

        batch_bboxes, batch_character_indices, batch_scores = (
            self.get_batch_bboxes_and_batch_character_indices(
                predicted_bboxes=predicted_bboxes,
                predicted_class_scores=predicted_class_scores,
                original_image_sizes=original_image_sizes,
            )
        )

        crop_bboxes, crop_embeddings_for_batch = self.get_crop_embeddings_for_batch(
            images=images,
            batch_bboxes=batch_bboxes,
            batch_character_indices=batch_character_indices,
            move_to_device_fn=self.move_to_device,
        )

        character_scores = []

        for character_score, character_indices in zip(batch_scores, batch_character_indices):
            character_scores.append(torch.gather(character_score, 0, character_indices))

        image_bboxes = {}
        i = 0
        for image, bboxes in zip(images, crop_bboxes):
            crops = self.processor.crop_image_new(image, bboxes)
            image_bboxes[i] = crops
            i = i + 1

        return crop_bboxes, crop_embeddings_for_batch, image_bboxes, character_scores

    # Извлекает предсказанные токены объектов из выходных данных трансформера мб не стоит создавать отдельную функцию и просто включить в код
    def _get_predicted_obj_tokens(self, detection_transformer_output: ConditionalDetrModelOutput):
        return detection_transformer_output.last_hidden_state[:, : -self.num_non_obj_tokens]

    # Получает предсказанные оценки классов и ограничивающие рамки для объектов из выходных данных трансформера
    def _get_predicted_bboxes_and_classes(
        self,
        detection_transformer_output: ConditionalDetrModelOutput,
    ):
        # Извлекаем предсказанные токены объектов
        obj = self._get_predicted_obj_tokens(detection_transformer_output)
        # Получаем предсказанные оценки классов объектов
        predicted_class_scores = self.class_labels_classifier(obj)
        # Получаем ссылочные точки для предсказанных боксов
        reference = detection_transformer_output.reference_points[: -self.num_non_obj_tokens]  # type: ignore
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

    def get_crop_embeddings_for_batch(
        self, images, batch_character_indices, batch_bboxes, move_to_device_fn
    ):
        crop_bboxes = [
            batch_bboxes[i][batch_character_indices[i]] for i in range(len(batch_character_indices))
        ]
        crop_embeddings_for_batch = self.predict_crop_embeddings(
            images, crop_bboxes, move_to_device_fn
        )
        return crop_bboxes, crop_embeddings_for_batch

    # Обнаружение объектов с использованием трансформера - тут просто суют в трансформер, мб можно и заменить на сразу сунуть в тарнсфомрер
    def _get_detection_transformer_output(
        self, pixel_values: torch.FloatTensor, pixel_mask: Optional[torch.LongTensor] = None
    ) -> ConditionalDetrModelOutput:
        return self.detection_transformer(
            pixel_values=pixel_values, pixel_mask=pixel_mask, return_dict=True
        )

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
        #! ATTENTION!!! ВАЖНО!!! - тут список где абсолютно все bboxes, то есть они не разделяются по каждому отдельному изображению
        #! они все вместе в списке в виде numpy arrays
        # Перемещение обрезанных изображений на устройство
        crops_per_image = move_to_device_fn(crops_per_image)

        # Обработка обрезанных bbox пакетами, чтобы избежать переполнения памяти
        embeddings = []
        for i in range(0, len(crops_per_image), batch_size):
            crops = crops_per_image[i : i + batch_size]
            # Получение встраиваемых эмбеддингов для каждого пакета
            embeddings_per_batch = self.crop_embedding_model(crops).last_hidden_state[:, 0]
            embeddings.append(embeddings_per_batch)
        embeddings = torch.cat(embeddings, dim=0)

        crop_embeddings_for_batch = []
        # Распределение полученных эмбеддингов по изображениям
        #! СВЕРНУЛИ ВСЕ ОБРАТНО ПО СТРАНИЦАМ
        for num_crops in num_crops_per_batch:
            crop_embeddings_for_batch.append(embeddings[:num_crops])
            embeddings = embeddings[num_crops:]

        # Восстановление маскировочного коэффициента на значение по умолчанию
        # тут они возвращают какую-то константу, которая по умолчанию стояла в трансформере
        self.crop_embedding_model.embeddings.config.mask_ratio = old_mask_ratio

        return crop_embeddings_for_batch
