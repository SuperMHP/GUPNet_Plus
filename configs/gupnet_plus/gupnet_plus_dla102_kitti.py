_base_ = [
    './gupnet_plus_dla34_kitti.py'
]
model = dict(
    backbone=dict(
        depth=102,
        with_identity_root=True,
        init_cfg=dict(checkpoint='http://dl.yf.io/dla/models/imagenet/dla102-d94d9790.pth')
        ),
    neck=dict(
        channels=[128, 256, 512, 1024]),
    bbox_head=dict(feat_channels=133,
                   in_channels=133,
    )
)