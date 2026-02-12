# GUPNet++

This is the official implementation of "GUPNet++: Geometry Uncertainty Propagation Network for Monocular 3D Object Detection".

<img src="resources/gupnet++.png" alt="vis2" style="zoom:100%;" />

## citation

If you find our work useful in your research, please consider citing:

    @article{lu2024gupnet++,
    title={Gupnet++: geometry uncertainty propagation network for monocular 3D object detection},
    author={Lu, Yan and Ma, Xinzhu and Yang, Lei and Zhang, Tianzhu and Liu, Yating and Chu, Qi and He, Tong and Li, Yonghui and Ouyang,  Wanli},
    journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
    year={2024},
    publisher={IEEE}
    }
    @article{lu2021geometry,
    title={Geometry Uncertainty Projection Network for Monocular 3D Object Detection},
    author={Lu, Yan and Ma, Xinzhu and Yang, Lei and Zhang, Tianzhu and Liu, Yating and Chu, Qi and Yan, Junjie and Ouyang, Wanli},
    journal={arXiv preprint arXiv:2107.13774},year={2021}}

## Usage 

### Installation

This project is based on [mmdetection3d repository](https://github.com/open-mmlab/mmdetection3d). You can refer to the original mmdetection3d README to install the requirements [English](MMDET_README.md) | [简体中文](MMDET_README_zh-CN.md). Here we provide our accurate steps corresponding to our experiment environments with specific version packages:

1. install mmcv

        pip install mmcv-full==1.6.0

2. install mmdetection

        git clone https://github.com/open-mmlab/mmdetection.git
        cd mmdetection
        git checkout v2.24.0  # switch to v2.24.0 branch (2.25.0 is also ok)
        pip install -r requirements/build.txt
        pip install -v -e .  #
        cd ..

3. install mmsegmentaion

        pip install mmsegmentation==0.26.0

4. install mmdetection3d (current repo)

        git clone https://github.com/SuperMHP/GUPNet_Plus.git
        cd GUPNet_Plus
        pip install -v -e .

5. Downloading datasets. 

    [KITTI](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d), including left color images, camera calibration matrices and training labels. 

    [NuScenes](https://www.nuscenes.org/nuscenes#download), including Mini, Trainval, Test of Full dataset (v1.0).

6. Putting the datasets as following directory

    updating in recent days.

### Train

KITTI training for evaluation set

    # PyTorch DDP
    CUDA_VISIBLE_DEVICES=0,1,2,3 bash tools/dist_train.sh configs/gupnet_plus/gupnet_plus_dla34_kitti.py

    # Slurm
    GPUS=4 GPUS_PER_NODE=4 bash tools/slurm_train.sh YOUR_PARTITION_NAME configs/gupnet_plus/gupnet_plus_dla34_kitti.py

KITTI training for test set 

    # PyTorch DDP
    CUDA_VISIBLE_DEVICES=0,1,2,3 bash tools/dist_train.sh configs/gupnet_plus/gupnet_plus_dla34_kitti_trainval.py

    # Slurm
    GPUS=4 GPUS_PER_NODE=4 bash tools/slurm_train.sh YOUR_PARTITION_NAME configs/gupnet_plus/gupnet_plus_dla34_kitti_trainval.py

NuScenes training for evaluation set (DLA34)

    # PyTorch DDP
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash tools/dist_train.sh configs/gupnet_plus/gupnet_plus_dla34_nuscenes.py

    # Slurm
    GPUS=8 GPUS_PER_NODE=8 bash tools/dist_train.sh configs/gupnet_plus/gupnet_plus_dla34_nuscenes.py
    
NuScenes training for evaluation set (HGLS104)

    # PyTorch DDP
    ## node 1
    MASTER_ADDR=YOUR_MASTER_ADDR NNODES=2 NODE_RANK=0 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash tools/dist_train.sh configs/gupnet_plus/gupnet_plus_hgls104_nuscenes.py
    ## node 2
    MASTER_ADDR=YOUR_MASTER_ADDR NNODES=2 NODE_RANK=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash tools/dist_train.sh configs/gupnet_plus/gupnet_plus_hgls104_nuscenes.py

    # Slurm
    GPUS=16 GPUS_PER_NODE=8 bash tools/dist_train.sh configs/gupnet_plus/gupnet_plus_hgls104_nuscenes.py

NuScenes training for test set (DLA34)

    coming soon

### Test

KITTI testing for evaluation set

    # PyTorch DDP
    CUDA_VISIBLE_DEVICES=0,1,2,3 bash tools/dist_test.sh configs/gupnet_plus/gupnet_plus_dla34_kitti.py XXXXXXX/your_model.pth --eval mAP

    # Slurm
    GPUS=4 GPUS_PER_NODE=4 bash tools/slurm_test.sh YOUR_PARTITION_NAME configs/gupnet_plus/gupnet_plus_dla34_kitti.py XXXXXXX/your_model.pth --eval mAP


KITTI testing for test set 

    # 1. PyTorch DDP
    CUDA_VISIBLE_DEVICES=0,1,2,3 bash tools/dist_test.sh configs/gupnet_plus/gupnet_plus_dla34_kitti_trainval.py XXXXXXX/your_model.pth --format-only --eval-options 'pklfile_prefix=results/kitti_results/' 'submission_prefix=results/kitti_results/'

    # 2. Slurm
    GPUS=4 GPUS_PER_NODE=4 bash bash tools/slurn_test.sh configs/gupnet_plus/gupnet_plus_dla34_kitti_trainval.py XXXXXXX/your_model.pth --format-only --eval-options 'pklfile_prefix=results/kitti_results/' 'submission_prefix=results/kitti_results/'

    # 3. zip files
    zip -r -j submit.zip results/kitti_results/img_bbox

    #4. submitting submit.zip on KITTI web

## Checkpoints

    coming soon

## ⚠️ Known Issues

#### An implementation bug about the β-NLL trick

**Description:**
In the 192 $th$-200 lines in the `mmdet3d/models/losses/uncertain_smooth_l1_loss.py`, we implement the β-NLL trick as follows:

    loss_bbox = torch.exp(self.beta*sigma.detach()) * (self.loss_weight)**(-self.beta) * \
            (self.loss_weight * uncertain_l1_loss(
            pred,
            target,
            weight,
            sigma=sigma,
            alpha=self.alpha,
            reduction=reduction,
            avg_factor=avg_factor))

Due to our oversight, `reduction` still maintained the default setting of mmdetetcion as `'mean'`, leading to the actual behavior of the loss from the expected function: 

$$\mathcal{L}^{\text{expected}}_{\beta\text{-NLL}} = \mathbb{E}_{i} \left[ \underbrace{\left\lfloor \frac{\sigma_{i}}{\sqrt{2}} \right\rfloor^\beta}_{\text{sample-wise}} \left( \frac{\sqrt{2}}{\sigma_i} |\mu_i - gt_i| + \log \sigma_i \right) \right],$$

where $N$ is the batch size, transferred to:

$$\mathcal{L}^{\text{actual}}_{\beta\text{-NLL}} = \mathbb{E}_{i}\left[\underbrace{\mathbb{E}_{i} \left[ \left\lfloor \frac{\sigma_{i}}{\sqrt{2}} \right\rfloor^\beta\right]}_{\text{bacth-wise}}\left( \frac{\sqrt{2}}{\sigma_i} |\mu_i - gt_i| + \log \sigma_i \right)\right].$$
    
**Impact Analysis**:
* **Mechanism:**  This results in a batch-wise (group-wise) weighting mechanism rather than a strict sample-wise one.
* **Effectiveness:** Our ablation studies and final performance reported in the paper are based on this implementation, demonstrating its effectiveness.
* **Reasoning:** Consistent with the motivation in our paper, the goal of the β-NLL loss is to "control the trade-off between uncertainty-based learning and traditional L1 regression optimization" and to mitigate the negative effects when "dealing with datasets with higher variance."
* **Conclusion**: Although the weighting became batch-wise, our analysis suggests it still functions effectively as a coarse-grained regularizer. When a batch (or the majority of samples within it) exhibits high uncertainty (large $\sigma$), the averaged weighting coefficient decreases, globally down-weighting the loss for that batch. This effectively prevents the training from being dominated by unstable gradients while preserving the core benefit of the uncertainty modeling discussed in the original paper.

**Recommendation:**
* **For Strict Reproduction:** To reproduce the exact results reported in the paper, please use the original code (with the implicit mean reduction).
* **For New Research:** If you aim to use the strict sample-wise β-NLL, please set `reduction='none'`. However, please note that this setting differs from our original implementation and has not been experimentally verified by us.

**Acknowledgement:**
We thank **@dornd2000** for identifying this implementation detail and providing valuable insights.

## Contact

If you have any question about this project, please feel free to contact yan.lu1@sydney.edu.au or luyan@pjlab.org.cn.
