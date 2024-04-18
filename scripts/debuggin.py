from pathlib import Path

import numpy as np
import torch
from libs.colouranga.from_magi_model import MyMagiModel
from libs.colouranga.from_magi_model.config import MagiConfig
from libs.colouranga.from_magi_model.utils import read_image_as_np_array as read_image
from transformers.modeling_utils import load_state_dict

images = [read_image(str(image)) for image in Path("data/manga2/").glob("*.jpg")]

chapter = np.vstack(images)
state_dict = load_state_dict(str(Path("models/magi/pytorch_model.bin").resolve()))
config: MagiConfig = MagiConfig.from_json_file(Path("libs/lizi/my_magi/config.json").resolve())  # type: ignore
model = MyMagiModel(config)
model.load_state_dict(state_dict, strict=False)
model.cuda()  # type: ignore

with torch.no_grad():
    results = model.predict_detections_and_associations([chapter])
