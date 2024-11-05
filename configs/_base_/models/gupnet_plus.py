_base_ = './gupnet.py'
# model settings
model = dict(
    neck=dict(
        use_dcn=True), 
    bbox_head=dict(
        clsmap_type = 'soft',
        basedim3d_type = 'soft',
        uncertainty_type = 'GeU++',
        group_reg_dims=(2, 4, 2, 2, 6, 24),  #offset_2d, size_2d, depth, offset_3d, size_3d, heading, up_point, low_point, x3d
        loss_size2d = dict(type='UncertainL1Loss', loss_weight=1.4142, alpha=1/1.4142, beta=0.5),
        loss_size3d = dict(type='UncertainL1Loss', loss_weight=1.4142, alpha=1/1.4142, beta=0.5),
        loss_depth = dict(type='UncertainL1Loss', loss_weight=1.4142, alpha=1/1.4142, beta=0.5),
        use_3d_nms = True,
        use_iounc = True,),
)