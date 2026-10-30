# OpenEVA

<p align="center">
  <a href="https://github.com/SeeleAI/OpenEVA"><img alt="Visitors" src="https://visitor-badge.laobi.icu/badge?page_id=SeeleAI.OpenEVA&left_color=7B2CBF&right_color=39D353"></a>
  <a href="https://github.com/SeeleAI"><img alt="GitHub" src="https://img.shields.io/badge/GitHub-SeeleAI-7B2CBF?style=for-the-badge&logo=github&logoColor=white"></a>
  <a href="https://huggingface.co/SEELE-AI"><img alt="Hugging Face" src="https://img.shields.io/badge/Hugging%20Face-SEELE--AI-39D353?style=for-the-badge&logo=huggingface&logoColor=black"></a>
  <a href="https://www.seeles.ai/research/pages/EVA01"><img alt="Project Page" src="https://img.shields.io/badge/Project%20Page-EVA01-39D353?style=for-the-badge"></a>
  <a href="https://arxiv.org/abs/2605.16745"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2605.16745-7B2CBF?style=for-the-badge&logo=arxiv&logoColor=white"></a>
  <a href="https://x.com/SEELE_AI1125"><img alt="X" src="https://img.shields.io/badge/X-SEELE_AI1125-111111?style=for-the-badge&logo=x&logoColor=white"></a>
  <a href="https://www.linkedin.com/company/seeleai"><img alt="LinkedIn" src="https://img.shields.io/badge/LinkedIn-SEELE--AI-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white"></a>
</p>

<div align="center">
  <img src="EVA01/assets/branding/OpenEVA-logo.png" alt="OpenEVA" width="520">
  <br>
  <img src="EVA01/assets/branding/seele-logo.png" alt="SEELE" width="260">
</div>

OpenEVA is the community release hub for SEELE-AI's EVA series: native 3D foundation models for mesh understanding and generation.

## 🗞️ News

- 🟢 **2026-10-20** · OpenEVA opens the EVA01 Unified All-Task generation runtime for text-to-3D and image-to-3D inference.
- 🟢 **2026-06-16** · OpenEVA opens the EVA01 mesh-understanding release path with inference, evaluation, and gallery assets.
- 🟣 **2026-05-16** · The EVA01 technical report appears on arXiv: [2605.16745](https://arxiv.org/abs/2605.16745).

## 🧭 Overview

EVA treats 3D meshes as a first-class multimodal language. The current release lets models read object geometry, reason about appearance and structure, and generate native 3D representations from text or images. Editing is on the roadmap.

The current public package contains the final **EVA01 Unified All-Task** boundary: mesh understanding/QA, text-to-3D, and image-to-3D. The generation runtime preserves the Unified MoT architecture and the native cascade `SS → Shape-512 → Material-512`. Checkpoints will be hosted on Hugging Face.

## Highlights

- **Native 3D modality**: EVA01 represents mesh input through a dedicated UND encoder and connector rather than through 2D image-only proxies.
- **Unified MoT generation**: one Qwen3-VL condition carrier drives the SS, Shape, and Material native branches with direct CUDA FlexAttention.
- **Two generation inputs**: the public generation entrypoint supports text-to-3D and single-image-to-3D and can export a decoded GLB mesh.

## 📦 Current Releases / Model Zoo

| Model | Status | Checkpoint | Notes |
| --- | --- | --- | --- |
| EVA01 Unified All-Task | 🟢 Generation adapter | `SEELE-AI/EVA01-Unified-All-Task-Generation` (Hugging Face link pending) | Generation-only Unified MoT adapter for text-to-3D and image-to-3D. Requires the matching frozen base assets. |
| EVA01 Unified All-Task LoRA | 🟢 Generation LoRA | `SEELE-AI/EVA01-Unified-All-Task-Generation-LoRA` (Hugging Face link pending) | Generation-only LoRA export with the same Unified MoT ABI. |
| EVA01-2B-Instruct | 🟢 Mesh understanding | [`SEELE-AI/EVA01-2B-Instruct`](https://huggingface.co/SEELE-AI/EVA01-2B-Instruct) | Mesh UND checkpoint for QA and description. |
| EVA01-2B-Instruct-LoRA | 🟢 Mesh understanding LoRA | [`SEELE-AI/EVA01-2B-Instruct-LoRA`](https://huggingface.co/SEELE-AI/EVA01-2B-Instruct-LoRA) | PEFT adapter release for mesh understanding. |
| EVA01 data | 🟣 Planned | TBA | Public data release is planned. |

## ⚡ Quick Start

```bash
cd EVA01
bash install.sh
source .venv/bin/activate
python infer.py --mesh assets/examples/construction_backhoe.glb
python generate.py --prompt "a small ceramic fox" --checkpoint /path/to/generation-only-adapter \
  --qwen-root /path/to/Qwen3-VL-2B-Instruct --ss-xl-stem /path/to/ss-xl \
  --shape-stem /path/to/shape-512 --material-stem /path/to/material-512 \
  --tokenizer-root /path/to/eva01-unified-tokenizer \
  --normalization-config /path/to/TRELLIS.2-4B/pipeline.json \
  --ss-decoder-stem /path/to/ss_dec_conv3d_16l8_fp16 \
  --output outputs/fox.npz
```

Add `--shape-decoder-stem`, `--material-decoder-stem`, and
`--output-glb outputs/fox.glb` to export a GLB with decoded geometry and a
vertex-color material preview. The SS decoder follows
the TRELLIS.2 `sparse_structure_decoder` entry and is supplied by
TRELLIS-image-large; Shape and Material use the TRELLIS.2 decoders. All three
decoder checkpoints remain external assets and are not bundled in OpenEVA. The
public writer preserves decoded geometry without optional topology repair or
remeshing.

The full install guide, Python API, Gradio app, evaluation commands, metrics, and gallery live in [EVA01/README.md](EVA01/README.md).

## 📐 Evaluation

EVA01 includes a one-command PointLLM-200 evaluation entrypoint. The deterministic path reports BLEU, ROUGE, METEOR, Sentence-BERT, and SimCSE under fixed seed `20260615`. Optional GPT-ref and GPT-img judge paths are available when API credentials and Blender PBR render assets are provided.

```bash
cd EVA01
python eval_pointllm200.py --variant full
python eval_pointllm200.py --variant lora
```

## 🖼️ Gallery

The EVA01 package gallery combines mesh-understanding examples with selected text-to-3D and image-to-3D examples. The understanding gallery includes 10 TexVerse `.glb` examples and matching 2x2 Blender PBR four-view previews; the generation gallery includes Blender-rendered previews and image-to-3D inputs. See the full gallery in [EVA01/README.md](EVA01/README.md).

➡️ See the rendered gallery in [EVA01/README.md](EVA01/README.md).

## 🗺️ Roadmap

- 🟣 **Data release**: public data packaging is planned.
- 🟡 **Training release**: training code and recipes are planned for a future release.
- 🟡 **Editing**: the edit branch and edit checkpoints are planned for a future release.
- 🧪 **Evaluation assets**: benchmark packaging and reproducibility evidence will continue to be consolidated around the public EVA01 flow.

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

## 📄 License

OpenEVA is released under the [Apache-2.0 license](LICENSE).

## Acknowledgements

OpenEVA bundles inference adapters and runtime glue around external model and
CUDA components. Those components remain separately licensed by their authors.
Users must review and comply with the license terms of every external asset
they download:

- Qwen3-VL, provided by Qwen.
- TRELLIS and TRELLIS.2 flow models and decoder implementations.
- O-Voxel, `flex_gemm`, `flash-attn`, PyTorch, Transformers, and related CUDA
  extensions.
- `trimesh`, Pillow, NumPy, and the other Python dependencies listed in
  `EVA01/pyproject.toml`.

The EVA01 and OpenEVA source files in this repository are released under the
Apache-2.0 license in [`LICENSE`](LICENSE). External checkpoints and runtime
extensions are not relicensed by this acknowledgement. Their upstream
repositories and package distributions are the authoritative sources for their
license texts and additional notices.

## 🔗 Contact

- 🌐 Project page: [seeles.ai/research/pages/EVA01](https://www.seeles.ai/research/pages/EVA01)
- 🤗 Hugging Face: [SEELE-AI](https://huggingface.co/SEELE-AI)
- 🧩 GitHub: [SeeleAI](https://github.com/SeeleAI)
- 💬 X: [SEELE_AI1125](https://x.com/SEELE_AI1125)
- 💼 LinkedIn: [SEELE-AI](https://www.linkedin.com/company/seeleai)
