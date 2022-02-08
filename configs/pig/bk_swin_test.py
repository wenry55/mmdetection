_base_ = '../swin/mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco.py'


dataset_type = 'CocoDataset'
data_root = '/raid/templates/data/pig-segm/'
classes = ('pig',)
pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth'  # noqa

dataset_type = 'CocoDataset'
data_root = '/raid/templates/data/pig-segm/'
classes = ('pig',)

fp16 = dict(loss_scale=dict(init_scale=512))

model = dict(
    type='MaskRCNN',
    backbone=dict(
        depths=[2, 2, 18, 2],
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            num_classes=1),
        mask_head=dict(
            type='FCNMaskHead',
            num_classes=1)),
    neck=dict(in_channels=[96, 192, 384, 768]))

data = dict(
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'annotations/instances_Train.json',
        img_prefix=data_root + 'images/',
        ),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'annotations/instances_Validation.json',
        img_prefix=data_root + 'images/',
        ),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'annotations/instances_Test.json',
        img_prefix=data_root + 'images/',
        ))


runner = dict(type='EpochBasedRunner', max_epochs=120)
