_base_ = [
    './gupnet_plus_dla34_nuscenes.py'
]
model = dict(
    backbone=dict(
        _delete_=True,
        type='HourglassNet',
        downsample_times=5,
        num_stacks=2,
        stage_channels=[256, 256, 384, 384, 384, 512],
        stage_blocks=[2, 2, 2, 2, 2, 4],
        norm_cfg=dict(type='BN', requires_grad=True)),
    neck=None,
    bbox_head=dict(feat_channels=256+2+10,
                   attr_branch=(256, ),
                   loss_attr = dict(type='CrossEntropyLoss', loss_weight=1.0))
)
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=16)