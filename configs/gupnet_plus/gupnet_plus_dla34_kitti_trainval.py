_base_ = [
    './gupnet_plus_dla34_kitti.py'
]
evaluation = dict(interval=1000)
disk_root = '/ailab/group/pjlab-ai4s/ai4astro/move_4.24/Datasets/3d_det/kitti/'
data = dict(
    train=dict(
        ann_file=disk_root + 'kitti_infos_trainval_mono3d.coco.json',
        info_file=disk_root + 'kitti_infos_trainval.pkl'),
    val=dict(
        ann_file=disk_root + 'kitti_infos_test_mono3d.coco.json',
        info_file=disk_root + 'kitti_infos_test.pkl'),
    test=dict(
        ann_file=disk_root + 'kitti_infos_test_mono3d.coco.json',
        info_file=disk_root + 'kitti_infos_test.pkl')
)