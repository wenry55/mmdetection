import json
import argparse
import mmcv
import os

def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert vgg (via) format to coco format')
    parser.add_argument('vgg_json', help='The path of vgg json')

    args = parser.parse_args()
    return args

def create_coco_obj():
    return {
        "images": [],
        "annotations": [],
        "categories": []
    }

def create_coco_img():
    return {
        "id": -1,
        "width": -1,
        "height": -1,
        "file_name": ''
    }

def create_coco_anno():
    return {
    "id": -1,
    "image_id": -1,
    "category_id": -1,
    "segmentation": [], # RLE or [polygon],
    "area": -1.0, # float
    "bbox": [], # [x,y,width,height],
    "iscrowd": 0 # 0 or 1,
}


def convert_vgg(vgg_json):
    coco_obj = create_coco_obj()

    for item in vgg_json:
        # print(item)
        print(vgg_json[item])

    pass

import os.path as osp

def convert_balloon_to_coco(ann_file, out_file, image_prefix):
    data_infos = mmcv.load(ann_file)

    annotations = []
    images = []
    obj_count = 0
    for idx, v in enumerate(mmcv.track_iter_progress(data_infos.values())):
        filename = v['filename']
        img_path = osp.join(image_prefix, filename)
        height, width = mmcv.imread(img_path).shape[:2]

        images.append(dict(
            id=idx,
            file_name=filename,
            height=height,
            width=width))

        bboxes = []
        labels = []
        masks = []

        #for _, obj in v['regions'].items():
        for obj in v['regions']:
            # assert not obj['region_attributes']
            obj = obj['shape_attributes']
            px = obj['all_points_x']
            py = obj['all_points_y']
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            x_min, y_min, x_max, y_max = (
                min(px), min(py), max(px), max(py))


            data_anno = dict(
                image_id=idx,
                id=obj_count,
                category_id=0,
                bbox=[x_min, y_min, x_max - x_min, y_max - y_min],
                area=(x_max - x_min) * (y_max - y_min),
                segmentation=[poly],
                iscrowd=0)
            annotations.append(data_anno)
            obj_count += 1

    coco_format_json = dict(
        images=images,
        annotations=annotations,
        categories=[{'id':0, 'name': 'Pig'}])
    mmcv.dump(coco_format_json, out_file)


def main():
    args = parse_args()

    # with open(f"{args.vgg_json}") as file:
        # Load its content and make a new dictionary
        #data = json.load(file)
        #convert_vgg(data)
        #convert_balloon_to_coco(args.vgg_json)

    convert_balloon_to_coco(args.vgg_json, 'out.json', os.path.dirname(args.vgg_json))
if __name__ == '__main__':
    main()
