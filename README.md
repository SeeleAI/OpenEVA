# EVA

EVA is a family of 3D foundation models for native object understanding and generation.

This community repository currently releases **EVA01**, a 2B instruction-tuned mesh understanding model that accepts a `.glb` mesh and answers questions about its geometry, appearance, parts, materials, and likely function.

Start here:

```bash
cd EVA01
bash install.sh
source .venv/bin/activate
python infer.py --mesh assets/examples/construction_backhoe.glb
```

See [EVA01/README.md](EVA01/README.md) for the model API, Gradio app, PointLLM-200 evaluation, reproduced metrics, and example gallery.
