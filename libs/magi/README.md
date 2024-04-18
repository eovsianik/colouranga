---
language:
- en
tags:
- Manga
- Object Detection
- OCR
- Clustering
- Diarisation
---
<style>
  .title-container {
    display: flex;
    flex-direction: column; /* Stack elements vertically */
    justify-content: center;
    align-items: center;
  }
  
  .title {
    font-size: 2em;
    text-align: center;
    color: #333;
    font-family: 'Comic Sans MS', cursive; /* Use Comic Sans MS font */
    text-transform: uppercase;
    letter-spacing: 0.1em;
    padding: 0.5em 0 0.2em;
    background: transparent;
  }
  
  .title span {
    background: -webkit-linear-gradient(45deg, #6495ED, #4169E1); /* Blue gradient */
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }

  .subheading {
    font-size: 1.5em; /* Adjust the size as needed */
    text-align: center;
    color: #555; /* Adjust the color as needed */
    font-family: 'Comic Sans MS', cursive; /* Use Comic Sans MS font */
  }

  .authors {
    font-size: 1em; /* Adjust the size as needed */
    text-align: center;
    color: #777; /* Adjust the color as needed */
    font-family: 'Comic Sans MS', cursive; /* Use Comic Sans MS font */
    padding-top: 1em;
  }

  .affil {
    font-size: 1em; /* Adjust the size as needed */
    text-align: center;
    color: #777; /* Adjust the color as needed */
    font-family: 'Comic Sans MS', cursive; /* Use Comic Sans MS font */
  }

</style>

<div class="title-container">
  <div class="title">
    The <span>Ma</span>n<span>g</span>a Wh<span>i</span>sperer
  </div>
  <div class="subheading">
    Automatically Generating Transcriptions for Comics
  </div>
  <div class="authors">
    Ragav Sachdeva and Andrew Zisserman
  </div>
  <div class="affil">
    University of Oxford
  </div>
  <div style="display: flex;">
    <a href="https://arxiv.org/abs/2401.10224"><img alt="Static Badge" src="https://img.shields.io/badge/arXiv-2401.10224-blue"></a>
    &emsp;
    <img alt="Dynamic JSON Badge" src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fmodels%2Fragavsachdeva%2Fmagi%3Fexpand%255B%255D%3Ddownloads%26expand%255B%255D%3DdownloadsAllTime&query=%24.downloadsAllTime&label=%F0%9F%A4%97%20Downloads">
  </div>
</div>

![image/png](https://cdn-uploads.huggingface.co/production/uploads/630852d2f0dc38fb47c347a4/B3ngZKXGZGBcZgPK6_XF0.png)

# Usage
```python
from transformers import AutoModel
import numpy as np
from PIL import Image
import torch
import os

images = [
        "path_to_image1.jpg",
        "path_to_image2.png",
    ]

def read_image_as_np_array(image_path):
    with open(image_path, "rb") as file:
        image = Image.open(file).convert("L").convert("RGB")
        image = np.array(image)
    return image

images = [read_image_as_np_array(image) for image in images]

model = AutoModel.from_pretrained("ragavsachdeva/magi", trust_remote_code=True).cuda()
with torch.no_grad():
    results = model.predict_detections_and_associations(images)
    text_bboxes_for_all_images = [x["texts"] for x in results]
    ocr_results = model.predict_ocr(images, text_bboxes_for_all_images)

for i in range(len(images)):
    model.visualise_single_image_prediction(images[i], results[i], filename=f"image_{i}.png")
    model.generate_transcript_for_single_image(results[i], ocr_results[i], filename=f"transcript_{i}.txt")
```

# License and Citation
The provided model and datasets are available for unrestricted use in personal, research, non-commercial, and not-for-profit endeavors. For any other usage scenarios, kindly contact me via email, providing a detailed description of your requirements, to establish a tailored licensing arrangement.
My contact information can be found on my website: ragavsachdeva [dot] github [dot] io

```
@misc{sachdeva2024manga,
      title={The Manga Whisperer: Automatically Generating Transcriptions for Comics}, 
      author={Ragav Sachdeva and Andrew Zisserman},
      year={2024},
      eprint={2401.10224},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```