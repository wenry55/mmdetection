# 다음의 설정은 로컬에서 학습한 모델을 기반드로 추가학습을 할려고 만든것임. 
# pretrained 가 work_dirs 아래에 있는 것을 사용함
_base_ = '../yolox/yolox_s_8x8_300e_coco.py'

# model settings
pretrained = '../../work_dirs/pig/bk_yolox/latest.pth'

model = dict(
    backbone=dict(deepen_factor=1.33, widen_factor=1.25,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(
        in_channels=[320, 640, 1280], out_channels=320, num_csp_blocks=4),
    bbox_head=dict(in_channels=320, feat_channels=320))

# customizations
data_root = '/raid/templates/data/pig-segm/'
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

max_epochs = 3000
runner = dict(type='EpochBasedRunner', max_epochs=3000)
