<p align="center">
<h1 align="center"><strong>N3D-VLM: Native 3D Grounding Enables
Accurate Spatial Reasoning in Vision-Language Models</strong></h1>
<!-- <h3 align="center">Arxiv 2025</h3> -->

<p align="center">
    <a href="https://w-ted.github.io/">Yuxin Wang</a><sup>1,2</sup>,
    <a href="https://www.kelei.site/">Lei Ke</a><sup>2</sup>,
    <a href="https://cyrilsterling.github.io/">Boqiang Zhang</a><sup>2</sup>,
    <a href="https://openreview.net/profile?id=~Tianyuan_Qu2">Tianyuan Qu</a><sup>2,3</sup>,
    <a href="https://hanxunyu.github.io/">Hanxun Yu</a><sup>2,4</sup>,
    <br>
    <a href="https://openreview.net/profile?id=~Zhenpeng_Huang1">Zhenpeng Huang</a><sup>2,5</sup>,
    <a href="https://raymond-myu.github.io/">Meng Yu</a><sup>2</sup>,
    <a href="https://www.danxurgb.net/">Dan Xu</a><sup>1✉️</sup>,
    <a href="https://sites.google.com/view/dongyu888/">Dong Yu</a><sup>2</sup>
    <br>
    <sup>1</sup>HKUST,
    <sup>2</sup>Tencent AI Lab,
    <sup>3</sup>CUHK,
    <sup>4</sup>ZJU,
    <sup>5</sup>NJU
</p>

<div align="center">
    <a href='https://arxiv.org/abs/2512.16561' target="_blank"><img src='https://img.shields.io/badge/arXiv-2512.16561-b31b1b.svg'></a>  
    <a href='https://n3d-vlm.github.io' target="_blank"><img src='https://img.shields.io/badge/Project-Page-Green'></a>  
    <a href='https://huggingface.co/yuxinhk/N3D-VLM' target="_blank">
        <img src='https://img.shields.io/badge/Hugging%20Face-Models-blue'>
    </a>
</div>
</p>




https://github.com/user-attachments/assets/077b6a9c-d1b3-4ebe-a9f1-e652b654b7d1


## Overview
**N3D-VLM** is a unified vision-language model for **native 3D grounding** and **3D spatial reasoning**. By incorporating native 3D grounding, our model enables precise spatial reasoning, allowing users to query object relationships, distances, and attributes directly within complex 3D environments.


## Updates

- **`2025/12/19`**: We released this repo with the pre-trained model and inference code.


## Installation

```
git clone --recursive https://github.com/W-Ted/N3D-VLM.git
cd N3D-VLM

conda env create -n n3d_vlm python=3.11 -y
conda activate n3d_vlm
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

## Pre-trained model
We provide the pre-trained model [here](https://huggingface.co/yuxinhk/N3D-VLM). 


## Inference 
We provide three examples for inference of N3D-VLM. You could check the source files in `data` directory, where `*.jpg` are the source images and `*.npz` are the monocular point clouds obtained by using [MoGe2](https://github.com/microsoft/moge). 
```
# inference 
python demo.py
```

### Demo 1


https://github.com/user-attachments/assets/e86306f2-152d-4337-a8d2-d165a26ce305


### Demo 2


https://github.com/user-attachments/assets/1bc0ee64-7a15-4592-941d-1037a26fb108


### Demo 3


https://github.com/user-attachments/assets/ba7ece12-4288-411d-9964-c676b78c6d5c


After running the code above, the inference results will be saved in the `outputs` directory, including generated answers in `*.json` format, and 3D grounding results in `*.rrd` format. 
The rrd files can be visualized by using [Rerun](https://rerun.io):
```
rerun outputs/demo1.rrd
```

If you want to do the 3D Detection only, please check the example as below. 
```
# inference 
python detection.py
# visualization
rerun outputs/test1.rrd
```

## Experiment 4.4.2 (ScanNet v2, real point cloud)

For thesis experiment **"3D semantic fusion and physical attribute extraction"**, you can run:

### 1) Prepare ScanNet data and GT boxes

```bash
python scripts/exp442_prepare_scannet.py \
    --scans_root /home/hba/Documents/Dataset/ScanNet/scans \
    --output_root outputs/exp442_scannet \
    --export_frame_npz \
    --frame_stride 60 \
    --max_frames_per_scene 20
```

This will generate:
- `outputs/exp442_scannet/manifest.json`
- per-scene `scene_points.npz`
- per-scene `gt_boxes.json` (AABB + PCA-OBB)
- optional per-frame `.npz` point clouds from real depth (no MoGe reconstruction)

### 2) Run benchmark (single-view vs multi-view+PCA/OBB)

```bash
python scripts/exp442_benchmark.py \
    --manifest outputs/exp442_scannet/manifest.json \
    --single_view_pred path/to/single_view_predictions.json \
    --multiview_pca_pred path/to/multiview_pca_predictions.json \
    --output_dir outputs/exp442_results
```

Output includes:
- `table_4_8.md` (3D-IoU AABB/OBB + centroid error)
- scene-level metric CSVs
- `summary.json` (with Wilcoxon significance)
- `figure_4_5_*.png` qualitative comparisons

### 3) Force demo to use real point cloud only

By default, `demo.py` now requires existing point cloud files. To allow fallback MoGe generation, explicitly add:

```bash
python demo.py --allow_moge_fallback
```

### Prediction JSON format

`exp442_benchmark.py` expects each prediction file as either:

```json
{
    "scene0000_00": [
        {"center": [x, y, z], "size": [sx, sy, sz], "yaw": 0.0, "label": "chair", "score": 0.9, "instance_id": "12"}
    ]
}
```

or

```json
[
    {"scene_id": "scene0000_00", "boxes": [{"center": [x, y, z], "size": [sx, sy, sz], "yaw": 0.0, "label": "chair"}]}
]
```


## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.


## Citation

```BibTeX
@article{wang2025n3d,
    title={N3D-VLM: Native 3D Grounding Enables Accurate Spatial Reasoning in Vision-Language Models},
    author={Wang, Yuxin and Ke, Lei and Zhang, Boqiang and Qu, Tianyuan and Yu, Hanxun and Huang, Zhenpeng and Yu, Meng and Xu, Dan and Yu, Dong},
    journal={arXiv preprint arXiv:2512.16561},
    year={2025}
}
```



