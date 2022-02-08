# The new config inherits a base config to highlight the necessary modification
_base_ = '../mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_coco.py'

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=1),
        mask_head=dict(num_classes=1)))

# Modify dataset related settings
# dataset_type = 'COCOPigDataset'
dataset_type = 'CocoPigDataset'
data_root = '/shared/data/pigs/'
classes = ('Pig',)
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'train/out.json',
        img_prefix=data_root + 'train/',),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'val/out.json',
        img_prefix=data_root + 'val/',
),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'val/out.json',
        img_prefix=data_root + 'val/',
))

# We can use the pre-trained Mask RCNN model to obtain higher performance
# load_from = 'checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'

runner = dict(type='EpochBasedRunner', max_epochs=96)