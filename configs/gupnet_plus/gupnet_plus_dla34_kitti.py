_base_ = [
    '../_base_/datasets/kitti-mono3d.py', '../_base_/models/gupnet_plus.py',
    '../_base_/schedules/mmdet_schedule_1x.py', '../_base_/default_runtime.py'
]
# model settings
find_unused_parameters = True
log_config = dict(interval=10)
checkpoint_config = dict(interval=10)
class_names = ['Pedestrian', 'Cyclist', 'Car']
model = dict(
    bbox_head=dict(num_classes=len(class_names),
                   feat_channels=66+len(class_names),
                   max_objs=50,
                   id2cat = class_names,
                   mean_size={
                        'Pedestrian': [0.84422524,1.76255119,0.66068622],
                        'Cyclist':[1.76282397,1.73698127,0.59706367],
                        'Car': [3.88311640418,1.52563191462,1.62856739989]           
                              },
                ),
)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFileMono3D'), 
    dict(
        type='LoadAnnotations3D',
        with_bbox=True,
        with_label=True,
        with_attr_label=False,
        with_bbox_3d=True,
        with_label_3d=True,
        with_bbox_depth=True,),
    dict(type='PhotoMetricDistortion'),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(type='RandomShiftScale', shift_scale=(0.1, 0.4), aug_prob=0.5),
    dict(type='AffineResize', img_scale=(1280, 384), down_ratio=4),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D',
        keys=[
            'img', 'gt_bboxes', 'gt_labels', 'gt_bboxes_3d',
            'gt_labels_3d', 'centers2d', 'depths',
        ]),
]
test_pipeline = [
    dict(type='LoadImageFromFileMono3D'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1280, 384),
        flip=False,
        transforms=[
            dict(type='AffineResize', img_scale=(1280, 384), down_ratio=4),
            dict(type='Normalize', **img_norm_cfg),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=6,
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))
# optimizer
optimizer = dict(_delete_=True, 
                 type='Adam', 
                 lr=0.00125, 
                 weight_decay=0.00001, 
                 paramwise_cfg=dict(bias_decay_mult=0.,
                                    custom_keys={'conv_offset': dict(lr_mult=0.1)}))
#optimizer_config = dict(
#    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    step=[90, 120])
total_epochs = 140
evaluation = dict(interval=10)
runner = dict(max_epochs=total_epochs)
