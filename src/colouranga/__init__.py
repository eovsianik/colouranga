from pathlib import Path

import huggingface_hub
import numpy as np
import torch
from anime_segmentation import get_model as get_anime_segmentation_model
from diffusers.schedulers import UniPCMultistepScheduler
from PIL import Image
from stable_diffusion_reference_only.pipelines.pipeline_stable_diffusion_reference_only import (
    StableDiffusionReferenceOnlyPipeline,
)

from colouranga.from_magi_model import MyMagiModel
from colouranga.from_magi_model.config import MagiConfig
from colouranga.utils import (
    character_segment,
    color_inversion,
    creating_pairs,
    finding_samples,
    get_embeddings,
    get_line_art,
    original_bboxes_compare,
    prepreparing_embeddings,
    sample_img,
    upload_pages,
)


class ColorizationPipeline:
    def __init__(self, device: str):
        self.device = device
        self.automatic_coloring_pipeline = StableDiffusionReferenceOnlyPipeline.from_pretrained(
            "AisingioroHao0/stable-diffusion-reference-only-automatic-coloring-0.1.2"
        ).to(device)
        self.automatic_coloring_pipeline.scheduler = UniPCMultistepScheduler.from_config(  # type: ignore
            self.automatic_coloring_pipeline.scheduler.config
        )
        self.segment_model = get_anime_segmentation_model(
            model_path=huggingface_hub.hf_hub_download("skytnt/anime-seg", "isnetis.ckpt")
        ).to(device)

        # Model initialization
        config: MagiConfig = MagiConfig.from_json_file(
            Path("src/colouranga/from_magi_model/config.json").resolve()
        )  # type: ignore
        self.model: MyMagiModel = MyMagiModel.from_pretrained("ragavsachdeva/magi", config=config)  # type: ignore
        self.model.to(device)  # type: ignore

    def colorization_pipeline(self, input_path: str, samples_path: str, output_path: str) -> None:
        """Colorizes all characters from manga pages.

        Args:
            input_path: directory that contains monochrome manga pages
            samples_path: directory that contains colorized samples of manga pages
            output_path: where to output colorized images of characters
        """
        my_pages = upload_pages(input_path)
        list_of_bboxes, original_emb = get_embeddings(self.model, my_pages)
        list_original_embeddings = prepreparing_embeddings(original_emb)
        my_comp_list = original_bboxes_compare(list_original_embeddings)
        my_crop_embeddings_sample, my_images_color_for_analysis = sample_img(
            self.model, samples_path
        )
        my_result_dict, my_sample_dict = finding_samples(my_crop_embeddings_sample, my_comp_list)
        final_character_list = creating_pairs(
            my_images_color_for_analysis, my_sample_dict, list_of_bboxes, my_result_dict
        )

        for elem in final_character_list:
            np_blue = elem.crop_image_bbox
            segmented_blue = character_segment(self.segment_model, np_blue)
            line_blue = get_line_art(segmented_blue)
            ready_blue = color_inversion(line_blue)

            colour_img = Image.open(elem.full_sample_file_name).convert("RGB")
            np_prompt = np.array(colour_img)
            ready_prompt = character_segment(self.segment_model, np_prompt)

            torch.cuda.empty_cache()
            ready_image = self.automatic_coloring_pipeline(
                prompt=Image.fromarray(ready_prompt),  # type: ignore
                blueprint=Image.fromarray(ready_blue),  # type: ignore
                num_inference_steps=20,
            )  # type: ignore
            new_filename = str(elem.crop_bboxes_coordinates)
            destination_path = Path(output_path) / new_filename

            ready_image.images[0].save(str(destination_path) + ".png")


def colorize_all_characters():
    raise NotImplementedError
