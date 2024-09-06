from typing import Any

from transformers import PretrainedConfig


class MagiConfig(PretrainedConfig):
    model_type = "magi"

    def __init__(
        self,
        detection_model_config: dict[str, Any],
        crop_embedding_model_config: dict[str, Any],
        detection_image_preprocessing_config: dict[str, Any],
        crop_embedding_image_preprocessing_config: dict[str, Any],
        **kwargs,
    ):
        self.detection_model_config = PretrainedConfig.from_dict(detection_model_config)
        self.crop_embedding_model_config = PretrainedConfig.from_dict(crop_embedding_model_config)
        self.detection_image_preprocessing_config = detection_image_preprocessing_config
        self.crop_embedding_image_preprocessing_config = crop_embedding_image_preprocessing_config
        super().__init__(**kwargs)
