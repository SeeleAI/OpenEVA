# EVA01: Unified Native 3D Understanding and Generation via Mixture-of-Transformers

<p align="center">
  <a href="https://github.com/SeeleAI/OpenEVA"><img alt="Visitors" src="https://visitor-badge.laobi.icu/badge?page_id=SeeleAI.OpenEVA.EVA01&left_color=7B2CBF&right_color=39D353"></a>
  <a href="https://github.com/SeeleAI/OpenEVA"><img alt="GitHub" src="https://img.shields.io/badge/GitHub-OpenEVA-7B2CBF?style=for-the-badge&logo=github&logoColor=white"></a>
  <a href="https://huggingface.co/SEELE-AI/EVA01-2B-Instruct"><img alt="Hugging Face Full" src="https://img.shields.io/badge/%F0%9F%A4%97%20Full-EVA01--2B--Instruct-39D353?style=for-the-badge"></a>
  <a href="https://huggingface.co/SEELE-AI/EVA01-2B-Instruct-LoRA"><img alt="Hugging Face LoRA" src="https://img.shields.io/badge/%F0%9F%A4%97%20LoRA-EVA01--2B--Instruct--LoRA-7B2CBF?style=for-the-badge"></a>
  <a href="https://www.seeles.ai/research/pages/EVA01"><img alt="Project Page" src="https://img.shields.io/badge/Project%20Page-EVA01-39D353?style=for-the-badge"></a>
  <a href="https://arxiv.org/abs/2605.16745"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2605.16745-7B2CBF?style=for-the-badge&logo=arxiv&logoColor=white"></a>
</p>

<p align="center">
  <img src="assets/teaser.png" alt="EVA01 teaser" width="92%">
</p>

EVA01 is a unified native 3D framework for mesh understanding, shape generation, and context-aware editing. The current OpenEVA release exposes the **UND-side** checkpoints and runtime: `.glb` mesh input is sampled deterministically, encoded by the EVA01 UND stack, projected into mesh tokens, and answered by a Qwen3-VL language backbone.

## 🌌 Overview

EVA01 integrates 3D meshes as a native modality for multimodal large language models. Its technical report describes a Mixture-of-Transformers architecture with a pre-trained Understanding Expert and a structurally mirrored Generation Expert, coupled through shared global self-attention and hard modality routing.

The public package in this repository is intentionally focused. It releases the UND-side model path, a compact inference interface, a Gradio chat app, PointLLM-200 evaluation, and 10 local TexVerse examples with real Blender PBR four-view renders.

## ✨ Release Status

| Item | Status |
| --- | --- |
| UND-side Full checkpoint | 🟢 [`SEELE-AI/EVA01-2B-Instruct`](https://huggingface.co/SEELE-AI/EVA01-2B-Instruct) |
| UND-side LoRA checkpoint | 🟢 [`SEELE-AI/EVA01-2B-Instruct-LoRA`](https://huggingface.co/SEELE-AI/EVA01-2B-Instruct-LoRA) |
| Training recipe | 🧪 Full and LoRA checkpoints are trained with alignment followed by instruction tuning. |
| Data release | 🟣 Planned. |
| Generation-side open source | 🟡 Planned for EVA02 rather than exposed in the current UND-side release. |

## ⚡ Install

The install script targets a clean Ubuntu-like machine. It bootstraps `uv`, creates `EVA01/.venv`, installs a current PyTorch wheel, and installs EVA01 in editable mode.

```bash
cd EVA01
bash install.sh
source .venv/bin/activate
```

If a different CUDA wheel index is required, set `TORCH_INDEX_URL` before running the script.

```bash
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121 bash install.sh
```

## 🚀 Quick Inference

```bash
python infer.py \
  --mesh assets/examples/construction_backhoe.glb \
  --question "Describe this 3D object in detail."
```

Python API:

```python
import torch
from eva01 import EVA01ForConditionalGeneration, EVA01Processor

model = EVA01ForConditionalGeneration.from_pretrained(
    "SEELE-AI/EVA01-2B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = EVA01Processor.from_pretrained("SEELE-AI/EVA01-2B-Instruct")

messages = [{
    "role": "user",
    "content": [
        {"type": "mesh", "mesh": "assets/examples/construction_backhoe.glb"},
        {"type": "text", "text": "Describe this 3D object in detail."},
    ],
}]

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

output_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False)
text = processor.batch_decode(
    output_ids[:, inputs.input_ids.shape[1]:],
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False,
)[0]
print(text)
```

The public mesh token is `<|mesh_und_pad|>`. The processor returns `input_ids`, `attention_mask`, and `mesh_und_values`.

## 💬 Gradio Chat

```bash
python app.py --host 127.0.0.1 --port 7860
```

The app loads the Full checkpoint by default, supports uploaded `.glb` files, and includes 10 built-in TexVerse examples.

## 📐 PointLLM-200 Evaluation

The eval script downloads PointLLM-200 benchmark files staged in the Full checkpoint repo and writes results under `EVA01/outputs/pointllm200/`.

```bash
python eval_pointllm200.py --variant full
python eval_pointllm200.py --variant lora
```

The deterministic path computes BLEU, ROUGE, METEOR, Sentence-BERT, and SimCSE with fixed seed `20260615` and greedy generation. Add `--with-gpt` to run GPT-ref and GPT-img when `OPENAI_API_KEY` is available.

```bash
OPENAI_API_KEY=... python eval_pointllm200.py --variant full --with-gpt
OPENAI_API_KEY=... python eval_pointllm200.py --variant lora --with-gpt
```

`OPENAI_JUDGE_MODEL` or `--gpt-model` selects the judge model. GPT-ref compares the model caption to the human caption. GPT-img judges the model caption against staged Blender PBR four-view renders produced from the original `.glb` assets under `benchmark/pointllm200/renders_blender_pbr/`. Use `--render-dir` to point to a different PBR render directory.

SimCSE loading with recent `transformers` releases requires `torch>=2.6` when the cached semantic model is stored as PyTorch `.bin` weights. The install script installs a current PyTorch wheel.

The PointLLM-200 benchmark source is cited as [`RunsenXu/PointLLM`](https://huggingface.co/datasets/RunsenXu/PointLLM).

## 🏁 Metrics

All metrics below are reported on PointLLM-200 with 200 samples. GPT-ref and GPT-img are judge metrics and may vary with judge model and API settings.

| Model | B-1 | B-4 | R-L | METEOR | SBERT | SimCSE | GPT-ref | GPT-img |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| EVA01-2B-Instruct | 6.386 | 0.589 | 9.443 | 13.505 | 50.651 | 50.767 | 59.045 | 70.335 |
| EVA01-2B-Instruct-LoRA | 6.372 | 0.607 | 9.455 | 13.567 | 51.194 | 51.320 | 59.560 | 71.480 |
| PointLLM-13B | 7.873 | 0.649 | 10.519 | 13.620 | 47.539 | 48.602 | 51.735 | 49.745 |
| ShapeLLM-13B | 10.542 | 1.050 | 12.954 | 14.234 | 39.935 | 40.728 | 33.925 | 35.870 |
| ShapeLLM-Omni | 11.326 | 1.197 | 14.190 | 13.276 | 34.617 | 35.115 | 25.625 | 20.190 |

The reproduced detailed semantic metrics are:

| Model | B-1 | B-2 | B-3 | B-4 | R-1 | R-2 | R-L | METEOR | SBERT | SimCSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| EVA01-2B-Instruct | 6.385873 | 2.313579 | 0.999180 | 0.589313 | 10.711027 | 1.628147 | 9.443079 | 13.505382 | 50.650976 | 50.766936 |
| EVA01-2B-Instruct-LoRA | 6.371676 | 2.289594 | 1.014568 | 0.606928 | 10.683788 | 1.604697 | 9.454826 | 13.567363 | 51.194399 | 51.319751 |

GPT-ref uses the original reference-judge run. GPT-img uses staged Blender PBR GLB renders in `benchmark/pointllm200/renders_blender_pbr/`.

## 🖼️ Gallery

The gallery uses 10 TexVerse GLB meshes included in `assets/examples/`. Each preview is a real Blender PBR 2x2 four-view render from the original `.glb`, ordered as front, right, back, and left. Full prompts, model answers, and GT captions are stored in `assets/examples/gallery.json`.

| Preview | Example | Q1 | A1 | Q2 | A2 | GT Caption |
| --- | --- | --- | --- | --- | --- | --- |
| <img src="assets/gallery/construction_backhoe_4view.png" alt="construction_backhoe PBR four-view render" width="180"> | `construction_backhoe` | Describe this 3D object in detail. | The object is a yellow construction vehicle with a large front bucket and a smaller rear bucket. It features a cabin with a windshield and a roof, an... | What work tools are mounted on the front and rear of this construction vehicle, and where are they attached? | The front and rear of the vehicle are equipped with bucket loaders, which are mounted on the front and rear for digging and lifti... | A yellow construction backhoe loader parked on a patch of dirt. The vehicle has a front loader bucket attached to a hydraulic arm, a rear digging arm folded behind, four... |
| <img src="assets/gallery/autumn_tree_4view.png" alt="autumn_tree PBR four-view render" width="180"> | `autumn_tree` | Describe this 3D object in detail. | The object is a tree with a thick, textured trunk and a dense canopy of yellow and orange leaves. The trunk is sturdy and has a natural, rough textur... | Which colors and structural details make this tree look like an autumn tree? | The tree has a brown trunk and branches, with leaves in shades of yellow, orange, and green, creating a vibrant autumn appearance. | A realistic 3D model of a deciduous tree in autumn. The tree consists of a thick, textured trunk at the center that branches out to support a dense canopy of leaves, whi... |
| <img src="assets/gallery/horse_head_sculpture_4view.png" alt="horse_head_sculpture PBR four-view render" width="180"> | `horse_head_sculpture` | Describe this 3D object in detail. | The object is a detailed model of a horse's head mounted on a rectangular base. The horse has a realistic texture with a brown coat and a white blaze... | What layers make up the base of this horse-head sculpture, and how do their materials differ? | The base is made of a smooth, light-colored material, contrasting with the textured, dark brown material of the horse's body. | A carved horse head sculpture mounted on a wooden base. The sculpture features the upper portion of a horse’s head attached to a round, dark pedestal, which is then fixe... |
| <img src="assets/gallery/patchwork_teddy_bear_4view.png" alt="patchwork_teddy_bear PBR four-view render" width="180"> | `patchwork_teddy_bear` | Describe this 3D object in detail. | The object is a colorful, stylized bear figure with a round head, small ears, and a snout. It has a vibrant, multicolored body with distinct sections... | What visible asymmetries appear between the left and right sides of this patchwork teddy bear? | The bear has a more pronounced asymmetry with the right side featuring a larger purple patch and a prominent red patch, while the... | A colorful patchwork teddy bear figurine. The bear has a rounded head with two ears, a small snout, and two mismatched button eyes, with its body, arms, and legs made up... |
| <img src="assets/gallery/ruined_japanese_house_4view.png" alt="ruined_japanese_house PBR four-view render" width="180"> | `ruined_japanese_house` | Describe this 3D object in detail. | The object is a rustic, weathered structure with a gabled roof and a series of windows. It appears to be made of wood and has a rough, aged texture.... | Which structural details suggest this wooden house is partly ruined and weather-exposed? | The presence of exposed beams, broken windows, and a partially collapsed roof suggests the house is partly ruined and weather-exp... | A traditional Japanese-style wooden house in a partially ruined state. The structure consists of a raised wooden foundation, open walls, a partially collapsed roof with... |
| <img src="assets/gallery/beige_backpack_4view.png" alt="beige_backpack PBR four-view render" width="180"> | `beige_backpack` | Describe this 3D object in detail. | The object is a backpack with a rugged, worn appearance. It features a main compartment with a flap closure and a front pocket. The backpack has mult... | What functional compartments and material details are visible on the front of this backpack? | The front of the backpack features a main compartment with a zipper, a side pocket with a flap, and a top pocket with a zipper. T... | A beige fabric backpack designed for casual or school use. The backpack has a large main compartment with a zipper closure on top, a smaller front pocket also with a zip... |
| <img src="assets/gallery/exposed_roof_framework_4view.png" alt="exposed_roof_framework PBR four-view render" width="180"> | `exposed_roof_framework` | Describe this 3D object in detail. | The object is a detailed architectural model of a building structure with a complex roof design. The roof features a combination of trusses and a cen... | Which beams, braces, posts, and chimney elements define the exposed roof framework? | The exposed roof framework is defined by beams, braces, posts, and a chimney. | A 3D model of a building structure with an exposed roof framework. The structure consists of a rectangular brick base wall with an open truss system above it, featuring... |
| <img src="assets/gallery/material_texture_grid_4view.png" alt="material_texture_grid PBR four-view render" width="180"> | `material_texture_grid` | Describe this 3D object in detail. | The object consists of six distinct panels, each with unique colors and textures. The panels are arranged in a grid-like formation, showcasing a vari... | How do the nine material tiles differ in surface shape, color, and apparent material? | The tiles differ in shape (rectangular, square, hexagonal), color (blue, green, orange, red, yellow, brown, gray), and apparent m... | A collection of nine square 3D material texture samples arranged in a 3x3 grid. The top row contains a silver metallic panel with irregular geometric cuts on the left, a... |
| <img src="assets/gallery/alien_plants_4view.png" alt="alien_plants PBR four-view render" width="180"> | `alien_plants` | Describe this 3D object in detail. | The object is a series of interconnected, colorful structures resembling a plant or a series of stacked cups. Each structure has a distinct color and... | What makes these plants look alien or fantastical? | The plants have exaggerated features such as large, bulbous heads, long stems, and vibrant colors, giving them an alien or fantas... | A collection of five fantastical alien plants. From left to right, the first plant has a dark brown root-like base with a glowing, curved, tentacle-shaped stem in bright... |
| <img src="assets/gallery/decorative_bottles_4view.png" alt="decorative_bottles PBR four-view render" width="180"> | `decorative_bottles` | Describe this 3D object in detail. | The object is a wooden shelf with a rectangular base and a flat top. It holds several bottles of various colors and shapes, including a tall bottle w... | How do the bottles differ in shape, color, and ornamentation? | The bottles differ in shape, color, and ornamentation, with some being tall and narrow, others round and bulbous, and some featur... | A collection of ornate decorative bottles displayed on a wooden shelf. The shelf holds six uniquely shaped bottles: on the top row, a purple bottle with twin handles and... |

## 📦 Checkpoint Format

Full checkpoints contain the Qwen3-VL weights under `qwen3vl/`, the EVA01 mesh UND encoder and connector in safetensors files, the tokenizer, `processor_config.json`, and `config.json`.

LoRA checkpoints contain standard PEFT adapter files plus the EVA01 mesh UND encoder and connector. The LoRA config points to `Qwen/Qwen3-VL-2B-Instruct` as the base model.

## 🗺️ Roadmap

- 🟣 Public data release is planned.
- 🟡 Generation-side open-source release is planned for EVA02 rather than exposed in the current UND-side release.
- 🧪 Benchmark packaging and reproducibility evidence will continue to follow the public EVA01 evaluation flow.

## 📚 Citation

```bibtex
@misc{eva01_2026,
  title  = {EVA01: Unified Native 3D Understanding and Generation via Mixture-of-Transformers},
  author = {Zongyuan Yang and Mingjing Yi and Wanli Ma and Chenzhuo Fan and Bocheng Li and Baolin Liu and Yuke Lou and Yingde Song and Yongping Xiong and Zhengdong Guo and Shimu Wang},
  year   = {2026},
  eprint = {2605.16745},
  archivePrefix = {arXiv},
  primaryClass = {cs.CV}
}
```
