_base_ = [
    '../_base_/datasets/nus-mono3d.py', '../_base_/models/gupnet_plus.py',
    '../_base_/schedules/mmdet_schedule_1x.py', '../_base_/default_runtime.py'
]
# model settings
find_unused_parameters = True
log_config = dict(interval=10)
checkpoint_config = dict(interval=10)
class_names = ['car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'barrier']
model = dict(
    bbox_head=dict(num_classes=len(class_names),
                   feat_channels=66+len(class_names),
                   max_objs=128,
                   id2cat = class_names,
                   bbox_code_size=9,
                   mean_size={
                        'pedestrian': [0.7298612286419041, 1.7614250283574295, 0.6707146972506797],
                        'barrier': [0.5, 0.98, 2.53],
                        'traffic_cone': [0.41, 1.07, 0.41],
                        'bicycle': [1.7, 1.28, 0.6],
                        'bus': [11.07388211506648, 3.46776974450095, 2.933345383248575],
                        'car': [4.62, 1.73, 1.95],
                        'construction_vehicle': [6.37, 3.19, 2.85],
                        'motorcycle': [2.11, 1.47, 0.77],
                        'trailer': [12.29, 3.87, 2.9],
                        'truck': [6.93, 2.84, 2.51]         
                              },)
)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFileMono3D'),
    dict(
        type='LoadAnnotations3D',
        with_bbox=True,
        with_label=True,
        with_attr_label=True,
        with_bbox_3d=True,
        with_label_3d=True,
        with_bbox_depth=True,
        with_kitti_addition=False),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(type='RandomShiftScale', shift_scale=(0.1, 0.4), aug_prob=0.5),
    dict(type='AffineResize', img_scale=(800,448), down_ratio=4),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D',
        keys=[
            'img', 'gt_bboxes', 'gt_labels', 'attr_labels', 'gt_bboxes_3d',
            'gt_labels_3d', 'centers2d', 'depths'
        ]),
]
test_pipeline = [
    dict(type='LoadImageFromFileMono3D'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(800,448),
        flip=False,
        transforms=[
            dict(type='AffineResize', img_scale=(800,448), down_ratio=4),
            dict(type='Normalize', **img_norm_cfg),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=8,
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline,
    samples_per_gpu=8))
# optimizer
optimizer = dict(_delete_=True, 
                 type='Adam', 
                 lr=0.00125, 
                 weight_decay=0.00001, 
                 paramwise_cfg=dict(bias_decay_mult=0.,
                                    custom_keys={'conv_offset': dict(lr_mult=0.1)}))
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
#learning policy
lr_config = dict(
    step=[90, 120])
total_epochs = 140

evaluation = dict(interval=10)
runner = dict(max_epochs=total_epochs)
