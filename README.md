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

### 1.1 Waymo Open Dataset (optional)

TGRIP can also be trained and evaluated on the [Waymo Open Dataset](https://waymo.com/open/) (Perception v2.0.1, parquet format). Only camera data is used.

**Download.** Accept the Waymo license with your Google account, install the [Google Cloud CLI](https://cloud.google.com/sdk/docs/install) and log in with `gcloud auth login --no-launch-browser`. Then download the camera-only components of the training and validation splits (~362 GB, of which ~361 GB are images):

```bash
cd /path/to/waymo
for s in training validation; do
  for c in camera_image camera_calibration vehicle_pose lidar_box camera_box camera_to_lidar_box_association; do
    mkdir -p "$s/$c"
    gcloud storage rsync -r "gs://waymo_open_dataset_v_2_0_1/$s/$c" "$s/$c"
  done
done
```

`lidar_box` contains the 3D box labels (not point clouds). The rsync is resumable. The `Is a directory` errors are harmless: they come from folder placeholder objects in the bucket.

Configure the path to the Waymo dataset in the [Makefile](./Makefile):

```bash
WAYMO_PATH = /path/to/waymo
```

**Preprocessing.** Inside the container, convert the dataset into a nuScenes-like index. Frames are subsampled from 10 Hz to 2 Hz (same 0.5 s step as nuScenes keyframes) and the camera JPEGs are copied as-is to `processed/` (~72 GB). `--verify` checks the camera geometry against Waymo's independent 2D labels at short (±15 m) and long (±50 m) range:

```bash
uv run tgrip/utils/preprocess_waymo.py --root ~/Datasets/waymo --split validation --workers 8
uv run tgrip/utils/preprocess_waymo.py --root ~/Datasets/waymo --split validation --verify
uv run tgrip/utils/preprocess_waymo.py --root ~/Datasets/waymo --split training --workers 8
uv run tgrip/utils/preprocess_waymo.py --root ~/Datasets/waymo --split training --verify
```

This gives 798 training / 202 validation segments, i.e. 27,092 training / 6,857 validation samples with the default temporal configuration.

**BEV semantic embeddings.** Generate the CLIP-L/14 visual embeddings of each object (~3 h, 2.1 GB), saved to the `semanticroot` of [waymo_pred.yaml](./configs/data/waymo_pred.yaml):

```bash
uv run tgrip/utils/create_semantic_gt_clip.py data=waymo_pred +clip_model=openai/clip-vit-large-patch14
```

As in nuScenes, they are only needed for training or to evaluate the cosine similarity; otherwise set `keep_input_semantic_maps: False`.

**Usage.** Select the Waymo data configuration, [waymo_pred.yaml](./configs/data/waymo_pred.yaml), in any task:

```bash
uv run tgrip/train.py data=waymo_pred
uv run tgrip/val.py data=waymo_pred
```

For long range evaluation, the optimized post-processing kernel is `model.postproc_kwargs.nms_kernel_size=5`.

**Differences with nuScenes:**

- 5 cameras (front, front-left, front-right, side-left, side-right); there is no rear camera. Front cameras (1920x1280) are cropped at the top (sky) to the size of the side cameras (1920x886) and all images are resized to 352x800 (native aspect ratio), see [waymo_scale_0_42.yaml](./configs/data/augs/waymo_scale_0_42.yaml).
- Waymo has no visibility annotation. Objects are used for training and evaluation only if they lie inside the field of view of at least one camera and have at least one lidar point in their box. Occlusion is not otherwise taken into account.
- There is a single vehicle class, mapped to `vehicle.car`. Mobility is derived from the box speed: `moving` above 0.5 m/s, otherwise `stopped`.
- Perspective segmentation, HD maps and lidar inputs are not supported.

## 2. Installation and Usage

[![CHANGELOG](https://img.shields.io/badge/Changelog-v1.1.0-2ea44f?style=for-the-badge)](https://github.com/miguelag99/TGRIP/blob/main/CHANGELOG.md)

The whole code is implemented inside a Docker image to ensure reproducibility and ease of use. The image is based on the official PyTorch image with CUDA support and includes all the necessary dependencies to run the code.
The project specific dependencies are installed using [uv](https://docs.astral.sh/uv/) in a virtual environment within the shared folder between the host and the container.

Before building the Docker image, you can configure the following parameters of the image in the [Makefile](./Makefile):

- `IMAGE_NAME`: Name of the generated Docker image.
- `TAG_NAME`: Tag of the generated Docker image.
- `USER_NAME`: Name of the user inside the Docker container.
- `NUSCENES_PATH`: Path to the NuScenes dataset (**MANDATORY**).
- `WAYMO_PATH`: Path to the Waymo dataset (optional, see [1.1 Waymo Open Dataset](#11-waymo-open-dataset-optional)).

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

### 2.2 Evaluation

To evaluate any version of TGRIP, you can use the following command inside the Docker container:

```bash
uv run tgrip/val.py
```

Remember to specify the model checkpoint to load in the [val.yaml](./configs/val.yaml) configuration file.

For prediction evaluation, it is not necessary to generate the BEV semantic embeddings for the whole dataset, as they are only used for training, just configure ```keep_input_semantic_maps: False``` in the [nuscenes_pred.yaml](./configs/data/nuscenes_pred.yaml) configuration file.

However, if you want to evaluate the model cosine similarity with the BEV semantic embeddings, you can generate them using the following command (only clip models and siglip have been tested):

```bash
uv run tgrip/utils/create_semantic_gt_<model>.py
```

You can change the model used inside each script. In create_semantic_gt_clip.py, it is selected with `+clip_model` (default `openai/clip-vit-base-patch32`), e.g. `+clip_model=openai/clip-vit-large-patch14` for CLIP-L/14.

## 3. Model checkpoints

The model checkpoints for the different versions of TGRIP are available in the [TGRIP GitHub repository](https://github.com/miguelag99/TGRIP/releases/tag/v1.1.0).

| Checkpoint Name | Download Link |
|-----------------|---------------|
| TGRIP_visual_CLIPL14.ckpt | [Download](https://github.com/miguelag99/TGRIP/releases/download/v1.1.0/TGRIP_visual_CLIPL14.ckpt) |
| TGRIP_visual_CLIPL14_short.ckpt | [Download](https://github.com/miguelag99/TGRIP/releases/download/v1.1.0/TGRIP_visual_CLIPL14_short.ckpt) |
| TGRIP_visual_CLIPB16.ckpt | [Download](https://github.com/miguelag99/TGRIP/releases/download/v1.1.0/TGRIP_visual_CLIPB16.ckpt) |

The configuration files default to CLIP-L/14. To use the CLIP-B/16 checkpoint, change:

| Setting | File | Value |
|---------|------|-------|
| `text_dim` | [semantic_conv.yaml](./configs/model/net/semantic_head/semantic_conv.yaml) | `512` |
| `semanticroot` | [nuscenes_pred.yaml](./configs/data/nuscenes_pred.yaml) | `${paths.data_dir}/visual_semantic_embeds_clipViTB16` |
| `model.text_encoder.model_name` | task config, e.g. [val.yaml](./configs/val.yaml) | `openai/clip-vit-base-patch16` |

## Citation
Please, consider citing this work with:

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
