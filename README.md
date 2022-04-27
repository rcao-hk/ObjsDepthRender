# ObjsDepthRender
Render synthetic depth maps for [preprocessed LineMOD](https://drive.google.com/drive/folders/19ivHpaKm9dOrr12fzC8IDFczWRPFxho7) and [GraspNet](https://graspnet.net/) datasets.

<img src="objsdepthrender.png" width="800px"/>

## Prerequisites

- Windows or Ubuntu Desktop (Offscreen rendering is waiting for update...)
- Python 3

## Requirements

- open3d
- numpy, h5py, opencv-python, scipy, matplotlib

## Usage

Generate rendered depth maps and RGB images in batches:
```shell
# for preprocessed linemod
python scene_render.py --dataset linemod --root_path $your_path_to_original_dataset_root --output_path $your_output_dir --output_width 640 --output_height 480

# for graspnet (realsense or kinect)
python scene_render.py --dataset graspnet --camera realsense --root_path $your_path_to_original_dataset_root --output_path $your_output_dir --output_width 1280 --output_height 720
```