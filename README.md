# TGRIP: Text Guided Representation for BEV Instance Prediction

<div align=center>
  <a href="https://github.com/miguelag99/BEVPredFormer/blob/main/CHANGELOG.md">
    <img src="https://img.shields.io/badge/Changelog-v1.0.0-2ea44f?style=for-the-badge" alt="CHANGELOG">
  </a>
  <a href="https://python.org">
    <img src="https://img.shields.io/badge/Python-3.12.3-3776AB.svg?style=for-the-badge&logo=python" alt="python">
  </a>
  <a href="https://pytorch.org">
    <img src="https://img.shields.io/badge/PyTorch-2.8.0-EE4C2C.svg?style=for-the-badge&logo=pytorch" alt="pytorch">
  </a>
  <a href="https://lightning.ai/docs/pytorch/stable/">
    <img src="https://img.shields.io/badge/Lightning-2.5.4-purple?style=for-the-badge&logo=lightning" alt="Lightning">
  </a>
</div>
<div align=center>
  <a href="https://wandb.ai/">
    <img src="https://img.shields.io/badge/Wandb-gray?style=for-the-badge&logo=weightsandbiases" alt="wandb">
  </a>
  <a href="https://www.docker.com">
    <img src="https://img.shields.io/badge/Docker-gray?style=for-the-badge&logo=docker&logoColor=white&labelColor=%23007FFF" alt="Docker">
  </a>
  <a href="https://docs.astral.sh/uv/">
    <img src="https://img.shields.io/badge/UV-gray?style=for-the-badge&logo=uv&logoColor=white&labelColor=DE5FE9" alt="UV">
  </a>
</div>

Implementation of TGRIP: Text Guided Representation for BEV Instance Prediction

## 1. NuScenes Dataset

Download the NuScenes dataset from the [official website](https://www.nuscenes.org/download) and extract the files in a folder with the following structure:

```bash
  nuscenes/
    ├──── maps/
    ├──── samples/
    ├──── sweeps/
    ├──── v1.0-trainval/
    └──── v1.0-mini/
```

Configure the path to the NuScenes dataset in the Makefile:

```bash
NUSCENES_PATH = /path/to/nuscenes
```

## 2. Installation and Usage

Build the Docker image with the following command:

```bash
make build
```

You can configure the following parameters of the image in the Makefile:

- `IMAGE_NAME`: Name of the generated Docker image.
- `TAG_NAME`: Tag of the generated Docker image.
- `USER_NAME`: Name of the user inside the Docker container.
- `NUSCENES_PATH`: Path to the NuScenes dataset.

Once the image is built, you can run the container with the following command:

```bash
make run
```

This command will run a bash inside the container and mount the current directory and dataset inside the container.
The launch script will automatically build the venv with requirements and CUDA ops the first time using `uv`.

### 2.1 Training

To train any version of BEVPredFormer, you can use the following command inside the Docker container:

```bash
uv run tgrip/train.py
```

The different configuration parameters can be tuned in the different yaml files located in the *configs* directory.

It is recommended to use some of the pretrained models available:

- BEVPredformer_Backbone_05.ckpt: Pretrained model with EfficientViT backbone for semantic segmentation (no prediction head). Recommended to use as a freezed backbone to train prediction models. Keys to load and freeze in train.yaml: `'net.backbone', 'net.neck', 'net.view_transform', 'net.decoder', 'net.coord_selector' and 'net.query_gen'`.

### 2.2 Map preprocessing

To preprocess the NuScenes maps, you can use the following command inside the Docker container:

```bash
uv run tgrip/preprocess_nuscenes_map.py --split val train --version=trainval
```

## Contact

[![Static Badge](https://img.shields.io/badge/ORCID-0009--0008--5627--5325-green?style=flat&logo=orcid)
](https://orcid.org/0009-0008-5627-5325)

If you have any questions, feel free to contact me at [miguel.antunes@uah.es](mailto:miguel.antunes@uah.es).
