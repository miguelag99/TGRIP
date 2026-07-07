# TGRIP: A Text-Guided Approach to Vehicle Instance Prediction in Autonomous Driving

<p align="center">
    <a href="https://www.miguelantunes.eu/">Miguel Antunes-García</a><sup>1</sup>,
    <a href="https://www.santimontiel.eu/">Santiago Montiel-Marín</a><sup>1</sup>,
    <a href="https://www.linkedin.com/in/fabio-sanchez-garcia/">Fabio Sánchez-García</a><sup>1</sup>,
</p>
<p align="center">
    <a href="https://rodrigogutierrezm.github.io/">Rodrigo Gutiérrez-Moreno</a><sup>1</sup>,
    <a href="https://scholar.google.es/citations?hl=es&user=IktmiSAAAAAJ">Rafael Barea</a><sup>1</sup>, and
    <a href="http://www.robesafe.uah.es/personal/bergasa/">Luis M. Bergasa</a><sup>1</sup>
</p>
<p align="center" style="font-size: 0.9em; font-style: italic;">
  <sup>1</sup> Universidad de Alcalá
</p>

<div align=center>
    <img src="https://img.shields.io/badge/Python-3.12.3-3776AB.svg?style=for-the-badge&logo=python" alt="python">
    <img src=https://img.shields.io/badge/PyTorch-2.8.0-EE4C2C.svg?style=for-the-badge&logo=pytorch>
    <img src=https://img.shields.io/badge/Lightning-2.5.4-purple?style=for-the-badge&logo=lightning>
</div>
<div align=center>
    <img src="https://img.shields.io/badge/UV-gray?style=for-the-badge&logo=uv&logoColor=white&labelColor=DE5FE9" alt="UV">
    <img src="https://img.shields.io/badge/Docker-gray?style=for-the-badge&logo=docker&logoColor=white&labelColor=%23007FFF" alt="Docker">
    <img src="https://img.shields.io/badge/Wandb-gray?style=for-the-badge&logo=weightsandbiases" alt="wandb">
    <a href="https://arxiv.org/abs/2607.04812">
      <img src="https://img.shields.io/badge/arxiv-black?style=for-the-badge&logo=arxiv" alt="arxiv">
    </a>
</div>

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

Configure the path to the NuScenes dataset in the [Makefile](./Makefile):

```bash
NUSCENES_PATH = /path/to/nuscenes
```

## 2. Installation and Usage

[![CHANGELOG](https://img.shields.io/badge/Changelog-v1.0.0-2ea44f?style=for-the-badge)](https://github.com/miguelag99/TGRIP/blob/main/CHANGELOG.md)

The whole code is implemented inside a Docker image to ensure reproducibility and ease of use. The image is based on the official PyTorch image with CUDA support and includes all the necessary dependencies to run the code.
The project specific dependencies are installed using [uv](https://docs.astral.sh/uv/) in a virtual environment within the shared folder between the host and the container.

Before building the Docker image, you can configure the following parameters of the image in the [Makefile](./Makefile):

- `IMAGE_NAME`: Name of the generated Docker image.
- `TAG_NAME`: Tag of the generated Docker image.
- `USER_NAME`: Name of the user inside the Docker container.
- `NUSCENES_PATH`: Path to the NuScenes dataset (**MANDATORY**).

Build the Docker image with the following command (requires make and Docker installed):

```bash
make build
```

Once the image is built, you can run the container with the following command:

```bash
make run
```

This command will run a bash inside the container and mount the current directory and dataset inside the container.
The launch script will automatically build the venv with requirements and CUDA ops.

### 2.1 Training

To train any version of TGRIP, you can use the following command inside the Docker container:

```bash
uv run tgrip/train.py
```

The different configuration parameters can be tuned in the different yaml files located in the [configs](./configs/) directory:

- [train.yaml](./configs/train.yaml): used to specify **checkpoint** to load, which parameters to freeze, training hyperparameters, resume training, etc.
- [nuscenes_pred.yaml](./configs/data/nuscenes_pred.yaml): used to specify preprocessing parameters for the NuScenes dataset (e.g., input image size, data augmentation, split, BEV grid configuration, etc.). It also contains the dataloading configuration (e.g., **batch size, number of workers**, etc.).
- [logger/default_pl.yaml](./configs/logger/default_pl.yaml): used to specify the **logger** configuration (e.g., Wandb project name, log directory, etc.).
- [trainer/ddp_pl.yaml](./configs/trainer/ddp_pl.yaml): used to specify the trainer configuration (e.g., **number of epochs, gpus, strategy** for multi-GPU training, etc.).
- [model/TGRIPPredictor.yaml](./configs/model/TGRIPPredictor.yaml): used to specify the **main model configuration** (e.g., model architecture, hyperparameters, etc.).

It is recommended to use some of the pretrained models available in the releases section of this repository to fine-tune the model for specific tasks. The available pretrained models are:

- TGRIP_visual_semantic.ckpt: full prediction model trained with visual semantic supervision from CLIP B/16 for the full BEV range of 50m from the ego-vehicle.
- TGRIP_visual_semantic_short.ckpt: full prediction model trained with visual semantic supervision from CLIP B/16 for short perception range of 30m.

### 2.2 Evaluation

To evaluate any version of TGRIP, you can use the following command inside the Docker container:

```bash
uv run tgrip/val.py
```

Remember to specify the model checkpoint to load in the [val.yaml](./configs/val.yaml) configuration file.

## 3. Model checkpoints

The model checkpoints for the different versions of TGRIP are available in the [TGRIP HuggingFace repository](https://huggingface.co/miguelag99/TGRIP).

| Semantic Supervision | IoU - Long | VPQ - Long | IoU - Short | VPQ - Long | Ckpt - Long | Ckpt - Short |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline (no semantic) | 40.9 | 33.3 | 63.9 | 54.9 | - | - |
| TGRIP CLIP-Base PS 16 | 41.3 | 34.3 | 64.5 | 56.1 | [Link](https://huggingface.co/miguelag99/TGRIP/resolve/main/TGRIP_visual_CLIPB16.ckpt) | [Link](https://huggingface.co/miguelag99/TGRIP/resolve/main/TGRIP_visual_CLIPB16_short.ckpt) |
| TGRIP CLIP-Large PS 14 | 41.3 | 34.3 | 64.5 | 56.3 | [Link](https://huggingface.co/miguelag99/TGRIP/resolve/main/TGRIP_visual_CLIPL14.ckpt) | [Link](https://huggingface.co/miguelag99/TGRIP/resolve/main/TGRIP_visual_CLIPL14_short.ckpt) |

## Citation
Please, consider citing thiw work with:

```bibtex
@misc{antunesgarcía2026tgriptextguidedapproachvehicle,
      title={TGRIP: A Text-Guided Approach to Vehicle Instance Prediction in Autonomous Driving}, 
      author={Miguel Antunes-García and Santiago Montiel-Marín and Fabio Sánchez-García and Rodrigo Gutiérrez-Moreno and Rafael Barea and Luis M. Bergasa},
      year={2026},
      eprint={2607.04812},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2607.04812}, 
}
```

## Contact

[![Static Badge](https://img.shields.io/badge/ORCID-0009--0008--5627--5325-green?style=flat&logo=orcid)](https://orcid.org/0009-0008-5627-5325)

If you have any questions, feel free to contact me at [miguel.antunes@uah.es](mailto:miguel.antunes@uah.es).
