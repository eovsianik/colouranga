import os
from dataclasses import dataclass

import networkx as nx
import numpy as np
import torch
from PIL import Image
from torchmetrics.functional.pairwise import pairwise_cosine_similarity

from colouranga.from_magi_model import MyMagiModel


# storing page image as a numpy array and its name
@dataclass
class ImageInfo:
    image: np.ndarray
    full_file_name: str

    def get_image_array(self):
        return self.image


# for CropBbox
@dataclass
class CropBbox:
    id_crop_bbox: int
    image_bbox: np.ndarray  # image of the bbox itself
    character_score: float
    embeddings_for_batch: torch.Tensor  # embedding for comparison
    crop_bboxes_for: torch.Tensor  # 4 coordinates
    file_name: str  # name of the original page


# example of the colored image and the name of the whole file with the image for coloring
@dataclass
class SampleImage:
    sample_image: np.ndarray  # image
    full_file_name: str  # name of the original page for later submission to coloring


# already transformed image with embedding for comparison
@dataclass
class AnalysisSampleImage:
    sample_image: np.ndarray  # image
    embeddings_for_batch: torch.Tensor  # embedding for comparison
    full_file_name: str  # name of the original page for later submission to coloring


@dataclass
class SampleImageConnection:
    crop_image_bbox: np.ndarray
    crop_bboxes_coordinates: torch.Tensor  # 4 coordinates
    file_page_name: str  # name of the original page, otherwise it won't match with coloring

    sample_image: np.ndarray  # example of colored image
    full_sample_file_name: str  # name of the original page for later submission to coloring


# Uploading images
def upload_pages(directory_path_uploading: str) -> list[ImageInfo]:
    pages_images = []
    for filename in os.listdir(directory_path_uploading):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            full_path = os.path.join(directory_path_uploading, filename)
            try:
                img = np.asarray(Image.open(full_path).convert("RGB"))
                pages_images.append(ImageInfo(image=img, full_file_name=full_path))
            except Exception as e:
                print(f"Error while opening {full_path}: {e}")
        else:
            print(f"Incorrect file extension {directory_path_uploading}")
    return pages_images


# get embeddings
def get_embeddings(
    model: MyMagiModel, images_pages: list[ImageInfo]
) -> tuple[list[CropBbox], list[torch.Tensor]]:
    images_for_everything = []
    list_of_embeddings = []
    id_number = 0
    for batch in images_pages:
        with torch.no_grad():
            page_image = [batch.image]
            page_name = batch.full_file_name

            (
                batch_crop_bboxes,
                batch_crop_embeddings_for_batch,
                batch_image_bboxes,
                batch_character_scores,
            ) = model.get_crops_and_embeddings(page_image)

        num_rows = len(batch_crop_embeddings_for_batch[0])

        for i in range(num_rows):
            images_for_everything.append(
                CropBbox(
                    id_crop_bbox=i + id_number,
                    image_bbox=batch_image_bboxes[0][i],
                    character_score=batch_character_scores[0][i],
                    embeddings_for_batch=batch_crop_embeddings_for_batch[0][i],
                    crop_bboxes_for=batch_crop_bboxes[0][i],
                    file_name=page_name,
                )
            )
            list_of_embeddings.append(batch_crop_embeddings_for_batch[0][i])

        id_number = id_number + num_rows
    return images_for_everything, list_of_embeddings


# preparing of embeddings
def prepreparing_embeddings(list_of_embeddings: list[torch.Tensor]) -> list[torch.Tensor]:
    sublists_for_embeddings = []
    for i in range(0, len(list_of_embeddings), 100):
        sublist_for_embeddings = list_of_embeddings[i : i + 100]
        sublists_for_embeddings.append(sublist_for_embeddings)

    # combine into tensors
    list_for_analysis = []
    for one_list in sublists_for_embeddings:
        crop_embeddings = None

        for i in range(len(one_list)):
            current_embeddings = one_list[i].unsqueeze(dim=0)

            if crop_embeddings is None:
                crop_embeddings = current_embeddings
            else:
                crop_embeddings = torch.cat((crop_embeddings, current_embeddings), dim=0)

        list_for_analysis.append(crop_embeddings)
    return list_for_analysis


# Finding maximum similarity
def original_bboxes_compare(list_for_analysis: list[torch.Tensor]) -> list[torch.Tensor]:
    compare_list = []
    for one_pack_for_analysis in list_for_analysis:
        pcs = pairwise_cosine_similarity(one_pack_for_analysis, one_pack_for_analysis)
        pcs = pcs.fill_diagonal_(0.0)
        new_var = torch.argmax(pcs, dim=1)
        char_to = torch.cat(
            (new_var.unsqueeze(1), torch.arange(len(new_var)).cuda().unsqueeze(1)), dim=1
        )
        graphs_chapter_one_max = nx.Graph(char_to.tolist())
        indices_per_chapter = [list(c_) for c_ in nx.connected_components(graphs_chapter_one_max)]
        for c_k in indices_per_chapter:
            for character_index in range(len(c_k)):
                num = int(c_k[character_index])
                if character_index == 0:
                    first_compare_batch = one_pack_for_analysis[num].unsqueeze(dim=0)
                else:
                    first_compare_batch = torch.cat(
                        (first_compare_batch, one_pack_for_analysis[num].unsqueeze(dim=0)), dim=0
                    )
            compare_list.append(first_compare_batch)
    return compare_list


# preparation for sample images
def sample_img(
    model: MyMagiModel,
    directory_path_samples: str,
) -> tuple[torch.Tensor, list[AnalysisSampleImage]]:
    images_samples = []
    for filename in os.listdir(directory_path_samples):
        full_path = os.path.join(directory_path_samples, filename)
        try:
            img = np.asarray(Image.open(full_path).convert("RGB"))
            images_samples.append(SampleImage(sample_image=img, full_file_name=full_path))
        except Exception as e:
            print(f"Error while opening {full_path}: {e}")

    images_color_for_analysis = []
    list_of_compare_embeddings = []

    for batch in images_samples:
        with torch.no_grad():
            page_image = [batch.sample_image]
            page_name = batch.full_file_name

            (
                batch_crop_bboxes,
                batch_crop_embeddings_for_batch,
                batch_image_bboxes,
                batch_character_scores,
            ) = model.get_crops_and_embeddings(page_image)

        num_rows = len(batch_crop_embeddings_for_batch[0])

        for i in range(num_rows):
            images_color_for_analysis.append(
                AnalysisSampleImage(
                    sample_image=batch_image_bboxes[0][i],
                    embeddings_for_batch=batch_crop_embeddings_for_batch[0][i],
                    full_file_name=page_name,
                )
            )
            list_of_compare_embeddings.append(batch_crop_embeddings_for_batch[0][i])

    crop_embeddings_sample = None

    for i in range(len(list_of_compare_embeddings)):
        current_embeddings = list_of_compare_embeddings[i].unsqueeze(dim=0)

        if crop_embeddings_sample is None:
            crop_embeddings_sample = current_embeddings
        else:
            crop_embeddings_sample = torch.cat(
                (crop_embeddings_sample, current_embeddings), dim=0
            )
    return crop_embeddings_sample, images_color_for_analysis # type: ignore


# match color examples embeddings and bboxes embeddings
def finding_samples(
    crop_embeddings_sample: torch.Tensor, compare_list: list[torch.Tensor]
) -> tuple[dict, dict]:
    result_dict = {}
    sample_dict = {}

    for one_tensor in compare_list:
        pcs_samples = pairwise_cosine_similarity(one_tensor, crop_embeddings_sample)
        comp = torch.sum(pcs_samples, dim=0)
        max_coincidence = int(torch.argmax(comp))
        if result_dict.get(max_coincidence) is not None:
            inter_res = torch.cat((result_dict[max_coincidence], one_tensor), dim=0)
            result_dict[max_coincidence] = inter_res
        else:
            result_dict[max_coincidence] = one_tensor
            sample_dict[max_coincidence] = crop_embeddings_sample[max_coincidence]
    return result_dict, sample_dict


# creating pairs
def creating_pairs(
    sample_list: list[AnalysisSampleImage],
    sample_dict: dict,
    original_list: list[CropBbox],
    original_dict: dict,
) -> list[SampleImageConnection]:
    list_for_colorization = []

    def compare_tensors(tensor1, tensor2):
        return torch.allclose(tensor1, tensor2)

    for key, tensor_value in sample_dict.items():
        matching_object: AnalysisSampleImage | None = next(
            (obj for obj in sample_list if compare_tensors(obj.embeddings_for_batch, tensor_value)),
            None,
        )
        if matching_object:
            picture_sample = matching_object.sample_image
            picture_name = matching_object.full_file_name

        for j in range(original_dict[key].shape[0]):
            matching_original: CropBbox | None = next(
                (
                    obj
                    for obj in original_list
                    if compare_tensors(obj.embeddings_for_batch, original_dict[key][j])
                ),
                None,
            )
            if matching_original:
                list_for_colorization.append(
                    SampleImageConnection(
                        crop_image_bbox=matching_original.image_bbox,
                        crop_bboxes_coordinates=matching_original.crop_bboxes_for,
                        file_page_name=matching_original.file_name,
                        sample_image=picture_sample,
                        full_sample_file_name=picture_name,
                    )
                )
    return list_for_colorization
