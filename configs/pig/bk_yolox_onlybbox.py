_base_ = '../yolox/yolox_s_8x8_300e_coco.py'

# model settings
# pretrained = 'https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_x_8x8_300e_coco/yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth'
# pretrained = './work_dirs/bk_yolox_onlybbox/latest.pth'
pretrained = '/home/bkseo_ext/git/mmdet220/mmdetection/work_dirs/bk_yolox_onlybbox/pre-4800.pth'
# pretrained = '/home/bkseo_ext/git/mmdet220/mmdetection/work_dirs/bk_yolox_onlybbox/epoch_1200_base.pth'
model = dict(
    backbone=dict(deepen_factor=1.33, widen_factor=1.25,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(
        in_channels=[320, 640, 1280], out_channels=320, num_csp_blocks=4),
    bbox_head=dict(in_channels=320, feat_channels=320))

# customizations
# data_root = '/raid/templates/data/pig-bbox/'
data_root = '/raid/templates/data/pig-bbox/single/'
#classes = ('pig','boar','sow', 'piglet', 'baby-pig')
classes = ('pig',)
dataset_type = 'CocoDataset'
model = dict(
    bbox_head=dict(
        type='YOLOXHead', num_classes=1, in_channels=128, feat_channels=128),)


train_dataset = dict(
    type='MultiImageMixDataset',
    dataset=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_Train.json',
        img_prefix=data_root + 'images/',
        classes=classes,
    ),)

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    persistent_workers=True,
    train=train_dataset,
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_Validation.json',
        img_prefix=data_root + 'images/',
        classes=classes,
        ),
    test=dict(
        type=dataset_type,
         ann_file=data_root + 'annotations/instances_Test.json',
        img_prefix=data_root + 'images/',
        classes=classes
        ))

runner = dict(type='EpochBasedRunner', max_epochs=9600)
