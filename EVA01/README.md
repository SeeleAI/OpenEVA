# EVA01

EVA01 is a 2B instruction-tuned model for native 3D mesh understanding. It accepts a `.glb` mesh, samples deterministic colored points, converts them into 513 mesh tokens with a frozen mesh UND encoder and connector, and answers with a Qwen3-VL language model.

Released checkpoints:

- Full: [`SEELE-AI/EVA01-2B-Instruct`](https://huggingface.co/SEELE-AI/EVA01-2B-Instruct)
- LoRA: [`SEELE-AI/EVA01-2B-Instruct-LoRA`](https://huggingface.co/SEELE-AI/EVA01-2B-Instruct-LoRA)

Both Full and LoRA are trained with alignment followed by instruction tuning.

## Install

The install script assumes a clean Ubuntu-like machine with Python available. It bootstraps `uv`, creates `EVA01/.venv`, installs PyTorch, and installs EVA01 in editable mode.

```bash
cd EVA01
bash install.sh
source .venv/bin/activate
```

If your CUDA wheel index differs from the default CUDA 12.4 wheel, set `TORCH_INDEX_URL` before running the script.

```bash
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121 bash install.sh
```

## Quick Inference

```bash
python infer.py \
  --mesh assets/examples/construction_backhoe.glb \
  --question "Describe this 3D object in detail."
```

Python API:

```python
import torch
from EVA01 import EVA01ForConditionalGeneration, EVA01Processor

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

## Gradio

```bash
python app.py --host 127.0.0.1 --port 7860
```

The app loads the Full model by default, supports uploaded `.glb` files, and includes 10 built-in TexVerse examples.

## PointLLM-200 Evaluation

The eval script downloads the PointLLM-200 benchmark files staged in the Full checkpoint repo and writes results under `EVA01/outputs/pointllm200/`.

```bash
python eval_pointllm200.py --variant full
python eval_pointllm200.py --variant lora
```

By default, the script computes BLEU, ROUGE, METEOR, Sentence-BERT, and SimCSE with fixed seed `20260615` and greedy generation. GPT judge scores are skipped unless you opt in:

SimCSE loading with recent `transformers` releases requires `torch>=2.6` when the cached semantic model is stored as PyTorch `.bin` weights. The install script requests a current PyTorch wheel.

```bash
OPENAI_API_KEY=... python eval_pointllm200.py --variant full --with-gpt
```

The PointLLM-200 benchmark source is cited as [`RunsenXu/PointLLM`](https://huggingface.co/datasets/RunsenXu/PointLLM).

## Metrics

All deterministic metrics below were reproduced on PointLLM-200 with 200 samples. GPT-ref and GPT-img are reference judge scores and are not bitwise reproducible because they depend on an external judge model and API settings.

| Model | B-1 | B-4 | R-L | METEOR | SBERT | SimCSE | GPT-ref | GPT-img |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| EVA01-2B-Instruct | 6.293 | 0.586 | 9.321 | 13.799 | 50.224 | 50.578 | N/A | N/A |
| EVA01-2B-Instruct-LoRA | 6.423 | 0.588 | 9.452 | 13.746 | 50.866 | 50.958 | N/A | N/A |
| PointLLM-13B | 7.873 | 0.649 | 10.519 | 13.620 | 47.539 | 48.602 | 51.735 | 49.745 |
| ShapeLLM-13B | 10.542 | 1.050 | 12.954 | 14.234 | 39.935 | 40.728 | 33.925 | 35.870 |
| ShapeLLM-Omni | 11.326 | 1.197 | 14.190 | 13.276 | 34.617 | 35.115 | 25.625 | 20.190 |

The reproduced detailed semantic metrics are:

GPT-ref and GPT-img for EVA01 were not filled in this checkout because the local API request to `api.openai.com/v1/responses` failed with `[Errno 101] Network is unreachable`. The public eval script keeps `--with-gpt` support so these judge metrics can be rerun in a network-enabled environment.

| Model | B-1 | B-2 | B-3 | B-4 | R-1 | R-2 | R-L | METEOR | SBERT | SimCSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| EVA01-2B-Instruct | 6.293355 | 2.323782 | 0.988426 | 0.586073 | 10.620402 | 1.630430 | 9.321463 | 13.799026 | 50.223697 | 50.577731 |
| EVA01-2B-Instruct-LoRA | 6.423183 | 2.255528 | 1.004870 | 0.588012 | 10.781779 | 1.577407 | 9.452272 | 13.745721 | 50.865533 | 50.957932 |

## Gallery

The gallery uses 10 TexVerse GLB meshes included in `assets/examples/`. The preview image is a 2x2 four-view colored point projection generated from the same mesh sampling path used by inference.

| Preview | Example | Q1 | A1 | Q2 | A2 | GT Caption |
| --- | --- | --- | --- | --- | --- | --- |
| ![construction_backhoe](assets/gallery/construction_backhoe_4view.png) | `construction_backhoe` | Describe this 3D object in detail. | The object is a yellow construction vehicle with a large front bucket and a smaller rear bucket. It features a cabin with a windshield and a roof, and it has four wheels. The vehicle is equipped with a front loader an... | What work tools are mounted on the front and rear of this construction vehicle, and where are they attached? | The front and rear of the vehicle are equipped with bucket loaders, which are mounted on the front and rear for digging and lifting materials. | A yellow construction backhoe loader parked on a patch of dirt. The vehicle has a front loader bucket attached to a hydraulic arm, a rear digging arm folded behind, four large rubber wheels, and an overhead canopy abo... |
| ![autumn_tree](assets/gallery/autumn_tree_4view.png) | `autumn_tree` | Describe this 3D object in detail. | The object is a tree with a thick, textured trunk and a dense canopy of yellow and orange leaves. The trunk is sturdy and has a natural, rough texture. The leaves are vibrant and appear to be in full bloom, creating a... | Which colors and structural details make this tree look like an autumn tree? | The tree has a brown trunk and branches, with leaves in shades of yellow, orange, and green, creating a vibrant autumn appearance. | A realistic 3D model of a deciduous tree in autumn. The tree consists of a thick, textured trunk at the center that branches out to support a dense canopy of leaves, which cascade downward in all directions. The trunk... |
| ![horse_head_sculpture](assets/gallery/horse_head_sculpture_4view.png) | `horse_head_sculpture` | Describe this 3D object in detail. | The object is a detailed model of a horse's head mounted on a rectangular base. The horse has a realistic texture with a brown coat and a white blaze running down its face. The head is sculpted with attention to detai... | What layers make up the base of this horse-head sculpture, and how do their materials differ? | The base is made of a smooth, light-colored material, contrasting with the textured, dark brown material of the horse's body. | A carved horse head sculpture mounted on a wooden base. The sculpture features the upper portion of a horse's head attached to a round, dark pedestal, which is then fixed onto a square light wooden block. The horse he... |
| ![patchwork_teddy_bear](assets/gallery/patchwork_teddy_bear_4view.png) | `patchwork_teddy_bear` | Describe this 3D object in detail. | The object is a colorful, stylized bear figure with a round head, small ears, and a snout. It has a vibrant, multicolored body with distinct sections in purple, green, blue, and red. The bear's limbs are rounded and m... | What visible asymmetries appear between the left and right sides of this patchwork teddy bear? | The bear has a more pronounced asymmetry with the right side featuring a larger purple patch and a prominent red patch, while the left side has a smaller green patch. | A colorful patchwork teddy bear figurine. The bear has a rounded head with two ears, a small snout, and two mismatched button eyes, with its body, arms, and legs made up of differently colored and textured sections ar... |
| ![ruined_japanese_house](assets/gallery/ruined_japanese_house_4view.png) | `ruined_japanese_house` | Describe this 3D object in detail. | The object is a rustic, weathered structure with a gabled roof and a series of windows. It appears to be made of wood and has a rough, aged texture. The structure is divided into two main sections, with one section fe... | Which structural details suggest this wooden house is partly ruined and weather-exposed? | The presence of exposed beams, broken windows, and a partially collapsed roof suggests the house is partly ruined and weather-exposed. | A traditional Japanese-style wooden house in a partially ruined state. The structure consists of a raised wooden foundation, open walls, a partially collapsed roof with wooden beams exposed, and a veranda extending al... |
| ![beige_backpack](assets/gallery/beige_backpack_4view.png) | `beige_backpack` | Describe this 3D object in detail. | The object is a backpack with a rugged, worn appearance. It features a main compartment with a flap closure and a front pocket. The backpack has multiple straps for carrying, including shoulder straps and a top handle... | What functional compartments and material details are visible on the front of this backpack? | The front of the backpack features a main compartment with a zipper, a side pocket with a flap, and a top pocket with a zipper. The material appears to be a durable fabric with a smooth texture. | A beige fabric backpack designed for casual or school use. The backpack has a large main compartment with a zipper closure on top, a smaller front pocket also with a zipper, and a brown leather patch with stitched det... |
| ![exposed_roof_framework](assets/gallery/exposed_roof_framework_4view.png) | `exposed_roof_framework` | Describe this 3D object in detail. | The object is a detailed architectural model of a building structure with a complex roof design. The roof features a combination of trusses and a central chimney. The walls are made of a textured material, and the roo... | Which beams, braces, posts, and chimney elements define the exposed roof framework? | The exposed roof framework is defined by beams, braces, posts, and a chimney. | A 3D model of a building structure with an exposed roof framework. The structure consists of a rectangular brick base wall with an open truss system above it, featuring angled beams, horizontal supports, vertical post... |
| ![material_texture_grid](assets/gallery/material_texture_grid_4view.png) | `material_texture_grid` | Describe this 3D object in detail. | The object consists of six distinct panels, each with unique colors and textures. The panels are arranged in a grid-like formation, showcasing a variety of patterns and materials. The panels include a mix of solid col... | How do the nine material tiles differ in surface shape, color, and apparent material? | The tiles differ in shape (rectangular, square, hexagonal), color (blue, green, orange, red, yellow, brown, gray), and apparent material (metallic, stone-like, wood-like, etc.). | A collection of nine square 3D material texture samples arranged in a 3x3 grid. The top row contains a silver metallic panel with irregular geometric cuts on the left, a gold pyramid-studded surface in the center, and... |
| ![alien_plants](assets/gallery/alien_plants_4view.png) | `alien_plants` | Describe this 3D object in detail. | The object is a series of interconnected, colorful structures resembling a plant or a series of stacked cups. Each structure has a distinct color and shape, with a combination of smooth and textured surfaces. The colo... | What makes these plants look alien or fantastical? | The plants have exaggerated features such as large, bulbous heads, long stems, and vibrant colors, giving them an alien or fantastical appearance. | A collection of five fantastical alien plants. From left to right, the first plant has a dark brown root-like base with a glowing, curved, tentacle-shaped stem in bright pink; the second plant features a tall green st... |
| ![decorative_bottles](assets/gallery/decorative_bottles_4view.png) | `decorative_bottles` | Describe this 3D object in detail. | The object is a wooden shelf with a rectangular base and a flat top. It holds several bottles of various colors and shapes, including a tall bottle with a cork, a round bottle with a cap, and a bottle with a narrow ne... | How do the bottles differ in shape, color, and ornamentation? | The bottles differ in shape, color, and ornamentation, with some being tall and narrow, others round and bulbous, and some featuring intricate patterns and designs. | A collection of ornate decorative bottles displayed on a wooden shelf. The shelf holds six uniquely shaped bottles: on the top row, a purple bottle with twin handles and a silver sphere cap is positioned on the left, ... |

## Checkpoint Format

Full checkpoints contain the Qwen3-VL weights under `qwen3vl/`, the EVA01 mesh UND encoder and connector in safetensors files, the tokenizer, `processor_config.json`, and `config.json`.

LoRA checkpoints contain standard PEFT adapter files plus the EVA01 mesh UND encoder and connector. The LoRA config points to `Qwen/Qwen3-VL-2B-Instruct` as the base model.
