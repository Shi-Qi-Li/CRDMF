# Multiple Rotation Averaging with Constrained Reweighting Deep Matrix Factorization

This is an official implementation of [*Multiple Rotation Averaging with Constrained Reweighting Deep Matrix Factorization*](https://ieeexplore.ieee.org/document/11128797) that is accepted to 2025 IEEE International Conference on Robotics & Automation (ICRA).

## Overview Video
[![Multiple Rotation Averaging with Constrained Reweighting Deep Matrix Factorization](https://i.ytimg.com/vi/g4_qnrJlu5Y/hqdefault.jpg?sqp=-oaymwFBCNACELwBSFryq4qpAzMIARUAAIhCGAHYAQHiAQoIGBACGAY4AUAB8AEB-AH-CYAC0AWKAgwIABABGGUgZShlMA8=&rs=AOn4CLB-FSfjz7fVKu1OLgdNDXPNgGNS8Q)](http://www.youtube.com/watch?v=g4_qnrJlu5Y&t "Multiple Rotation Averaging with Constrained Reweighting Deep Matrix Factorization")

## Installation
First, create the conda environment.
```
conda create -n crdmf python=3.9
conda activate crdmf
pip install -r requirements.txt
```
Then, install the [graph_ops](model/graph_ops/README.md) in `./model`.


## Data Preparation
The processed 1DSfM dataset can be found from [DMF-synch](https://github.com/gktejus/DMF-synch).

Please put all `.pt` and `.mat` files to `./data/1dsfm`.

## Run
Use the following command to run the experiment on one scenario.
```
python main.py --config config/1dsfm/Alamo.yaml
```
We also provide a script to run all experiments on 1DSfM.
```
sh run_1dsfm.sh
```

## Cite
If you find this code useful for your work, please consider citing:
```
@inproceedings{li2025multiple,
  title={Multiple Rotation Averaging with Constrained Reweighting Deep Matrix Factorization},
  author={Li, Shiqi and Zhu, Jihua and Xie, Yifan and Hu, Naiwen and Zhu, Mingchen and Li, Zhongyu and Wang, Di and Lu, Huimin},
  booktitle={2025 IEEE International Conference on Robotics and Automation (ICRA)},
  pages={16095--16101},
  year={2025},
  organization={IEEE}
}
```

## Acknowledgement
We thank the authors of the [DMF-synch](https://github.com/gktejus/DMF-synch), [HARA](https://github.com/sunghoon031/HARA) for open sourcing their codes.