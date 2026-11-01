# Aligned Delta-Triplane Transformers as Occupancy World Models

**NeurIPS 2026**

## Citation

If you find this work useful, please cite our paper. The BibTeX file is available at [citation.bib](citation.bib).

```bibtex
@inproceedings{xu2026aligned,
  title     = {Aligned Delta-Triplane Transformers as Occupancy World Models},
  author    = {Xu, Haoran and Peng, Peixi and Guo, Shijun and Tan, Guang and Chang, Yiqian and Zhao, Yisen and Wang, Shuaixian and Tian, Yonghong},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## TODO

- [ ] Release training code and configurations.
- [ ] Release testing and evaluation code and configurations.

Training and testing code will be made publicly available in a future update.

## 🛠️Environment

Install Pytorch 1.13 + CUDA 11.6

1. Create conda environment with python version 3.8.0

2. Install all the packages in environment.yaml

3. Anything about the installation of mmdetection3d, please refer to [mmdetection3d](https://mmdetection3d.readthedocs.io/en/latest/getting_started.html#installation)


## 📦 Prepare Dataset

1. Download nuScenes from [nuScenes](https://www.nuscenes.org/nuscenes)

2. Download Occ3D-nus from [Occ3D-nus](https://drive.google.com/file/d/1kiXVNSEi3UrNERPMz_CfiJXKkgts_5dY/view?usp=drive_link)

3. (Optional) Download Occ3D-Waymo from [Occ3D-Waymo](https://drive.google.com/drive/folders/13WxRl9Zb_AshEwvD96Uwz8cHjRNrtfQk) and unzip it to the `data/waymo` folder. We only use the validation of Occ3D-Waymo in our experiments.

4. (Optional) Download STCOcc-Res from [STCOcc-Res](https://drive.google.com/file/d/1dXB9mtROLWChycBZlhYIf_JBLshXogBs/view?usp=drive_link) and unzip it to the `data/nuscenes` folder.

5. Download train/val pickle files and put them in data/
    [nuscenes_infos_train_temporal_v3_scene.pkl](https://cloud.tsinghua.edu.cn/d/9e231ed16e4a4caca3bd/)
    [nuscenes_infos_val_temporal_v3_scene.pkl](https://cloud.tsinghua.edu.cn/d/9e231ed16e4a4caca3bd/)

## License

Original contributions to this repository are released under the [Apache License 2.0](LICENSE), which permits academic and commercial use subject to its terms.

Third-party code retains its original licenses. See [NOTICE](NOTICE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for attributions and included license texts. Datasets and externally obtained checkpoints remain subject to their providers' terms.

## Acknowledgement

Thanks to the following excellent projects:
- [GenieDrive](https://github.com/Huster-YZY/GenieDrive)
- [I2World](https://github.com/lzzzzzm/II-World)
- [OccWorld](https://github.com/wzzheng/OccWorld)
- [OccSora](https://github.com/wzzheng/OccSora)
- [mmdetection3d](https://github.com/open-mmlab/mmdetection3d)
