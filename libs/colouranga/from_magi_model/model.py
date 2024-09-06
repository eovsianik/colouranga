from typing import List, Optional

import torch
from libs.colouranga.from_magi_model.config import MagiConfig
from libs.colouranga.from_magi_model.processor import MagiProcessor
from libs.colouranga.from_magi_model.utils import (
    move_to_device,
    visualise_single_image_prediction,
)
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

        self.detection_transformer = ConditionalDetrModel(config.detection_model_config)  # type: ignore

        self.bbox_predictor = ConditionalDetrMLPPredictionHead(
            input_dim=config.detection_model_config.d_model,
            hidden_dim=config.detection_model_config.d_model,
            output_dim=4,
            num_layers=3,
        )

        self.character_character_matching_head = ConditionalDetrMLPPredictionHead(
            input_dim=3 * config.detection_model_config.d_model
            + (2 * config.crop_embedding_model_config.hidden_size),
            hidden_dim=config.detection_model_config.d_model,
            output_dim=1,
            num_layers=3,
        )

        self.class_labels_classifier = nn.Linear(
            config.detection_model_config.d_model, config.detection_model_config.num_labels
        )

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
        original_image_sizes,
        character_detection_threshold=0.3,
    ):
        batch_scores, batch_labels = predicted_class_scores.max(-1)

        batch_scores = batch_scores.sigmoid()
        batch_labels = batch_labels.long()

        batch_bboxes = center_to_corners_format(predicted_bboxes)  # type: ignore

        if isinstance(original_image_sizes, List):
            img_h = torch.Tensor([i[0] for i in original_image_sizes])
            img_w = torch.Tensor([i[1] for i in original_image_sizes])
        else:
            img_h, img_w = original_image_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(batch_bboxes.device)
        batch_bboxes: torch.Tensor = batch_bboxes * scale_fct[:, None, :]

        batch_character_indices = self.processor._get_indices_of_characters_to_keep(
            batch_scores, batch_labels, batch_bboxes, character_detection_threshold
        )

        return batch_bboxes, batch_character_indices, batch_scores

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

    def get_crops_and_embeddings(self, images: list[NDArray]):
        predicted_class_scores, predicted_bboxes = self.get_detr_output(images)

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

        image_bboxes = []

        for image, bboxes in zip(images, crop_bboxes):
            crops = self.processor.crop_image(image, bboxes)
            image_bboxes.append(crops)

        return crop_bboxes, crop_embeddings_for_batch, image_bboxes, character_scores

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

        def get_character_character_matching_scores(batch_character_indices, batch_bboxes):
            predicted_obj_tokens_for_batch = self._get_predicted_obj_tokens(
                detection_transformer_output
            )

            predicted_c2c_tokens_for_batch = self._get_predicted_c2c_tokens(
                detection_transformer_output
            )

            crop_bboxes = [
                batch_bboxes[i][batch_character_indices[i]]
                for i in range(len(batch_character_indices))
            ]

            crop_embeddings_for_batch = self.predict_crop_embeddings(
                images, crop_bboxes, move_to_device_fn
            )

            character_obj_tokens_for_batch = []
            c2c_tokens_for_batch = []

            for predicted_obj_tokens, predicted_c2c_tokens, character_indices in zip(
                predicted_obj_tokens_for_batch,
                predicted_c2c_tokens_for_batch,
                batch_character_indices,
            ):
                character_obj_tokens_for_batch.append(predicted_obj_tokens[character_indices])

                c2c_tokens_for_batch.append(predicted_c2c_tokens)

            return self._get_character_character_affinity_matrices(
                character_obj_tokens_for_batch=character_obj_tokens_for_batch,
                crop_embeddings_for_batch=crop_embeddings_for_batch,  # type: ignore
                c2c_tokens_for_batch=c2c_tokens_for_batch,
                apply_sigmoid=True,
            )

        return self.processor.postprocess_detections_and_associations(
            images=images,
            predicted_bboxes=predicted_bboxes,
            predicted_class_scores=predicted_class_scores,
            original_image_sizes=torch.stack(
                [torch.tensor(img.shape[:2]) for img in images], dim=0
            ).to(predicted_bboxes.device),
            get_character_character_matching_scores=get_character_character_matching_scores,
            character_detection_threshold=character_detection_threshold,
            panel_detection_threshold=panel_detection_threshold,
            text_detection_threshold=text_detection_threshold,
            character_character_matching_threshold=character_character_matching_threshold,
        )

    def predict_crop_embeddings(
        self, images, crop_bboxes, move_to_device_fn=None, mask_ratio=0.0, batch_size=256
    ):
        assert isinstance(
            crop_bboxes, List
        ), "please provide a list of bboxes for each image to get embeddings for"

        move_to_device_fn = self.move_to_device if move_to_device_fn is None else move_to_device_fn

        old_mask_ratio = self.crop_embedding_model.embeddings.config.mask_ratio
        self.crop_embedding_model.embeddings.config.mask_ratio = mask_ratio

        crops_per_image = []

        num_crops_per_batch = [len(bboxes) for bboxes in crop_bboxes]

        for image, bboxes, num_crops in zip(images, crop_bboxes, num_crops_per_batch):
            crops = self.processor.crop_image(image, bboxes)

            assert len(crops) == num_crops

            crops_per_image.extend(crops)

        if len(crops_per_image) == 0:
            return [[] for _ in crop_bboxes]

        crops_per_image = self.processor.preprocess_inputs_for_crop_embeddings(crops_per_image)

        crops_per_image = move_to_device_fn(crops_per_image)

        embeddings = []
        for i in range(0, len(crops_per_image), batch_size):
            crops = crops_per_image[i : i + batch_size]

            embeddings_per_batch = self.crop_embedding_model(crops).last_hidden_state[:, 0]
            embeddings.append(embeddings_per_batch)
        embeddings = torch.cat(embeddings, dim=0)

        crop_embeddings_for_batch = []

        for num_crops in num_crops_per_batch:
            crop_embeddings_for_batch.append(embeddings[:num_crops])
            embeddings = embeddings[num_crops:]

        self.crop_embedding_model.embeddings.config.mask_ratio = old_mask_ratio

        return crop_embeddings_for_batch

    def visualise_single_image_prediction(self, image_as_np_array, predictions, filename=None):
        return visualise_single_image_prediction(image_as_np_array, predictions, filename)

    def _get_detection_transformer_output(
        self, pixel_values: torch.FloatTensor, pixel_mask: Optional[torch.LongTensor] = None
    ) -> ConditionalDetrModelOutput:
        return self.detection_transformer(
            pixel_values=pixel_values, pixel_mask=pixel_mask, return_dict=True
        )

    def _get_predicted_obj_tokens(self, detection_transformer_output: ConditionalDetrModelOutput):
        return detection_transformer_output.last_hidden_state[:, : -self.num_non_obj_tokens]

    def _get_predicted_c2c_tokens(self, detection_transformer_output: ConditionalDetrModelOutput):
        return detection_transformer_output.last_hidden_state[:, -self.num_non_obj_tokens]

    def _get_predicted_bboxes_and_classes(
        self,
        detection_transformer_output: ConditionalDetrModelOutput,
    ):
        obj = self._get_predicted_obj_tokens(detection_transformer_output)

        predicted_class_scores = self.class_labels_classifier(obj)

        reference = detection_transformer_output.reference_points[: -self.num_non_obj_tokens]  # type: ignore

        reference_before_sigmoid = inverse_sigmoid(reference).transpose(0, 1)

        predicted_boxes = self.bbox_predictor(obj)

        predicted_boxes[..., :2] += reference_before_sigmoid

        predicted_boxes = predicted_boxes.sigmoid()

        return predicted_class_scores, predicted_boxes
