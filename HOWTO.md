## config 설정

https://mmcv.readthedocs.io/en/latest/understand_mmcv/config.html

## 사용모델 결정

성능평가 참고후 사용모델을 결정.

https://github.com/open-mmlab/mmdetection/blob/master/configs/yolox/README.md#results-and-models

configs/yolox/yolox_x_8x8_300e_coco.py

## 설정 변경

## 학습
```
python tools/train.py 
usage: train.py [-h] [--work-dir WORK_DIR] [--resume-from RESUME_FROM]  
                [--no-validate]  
                [--gpus GPUS | --gpu-ids GPU_IDS [GPU_IDS ...]] [--seed SEED]  
                [--deterministic] [--options OPTIONS [OPTIONS ...]]  
                [--cfg-options CFG_OPTIONS [CFG_OPTIONS ...]]  
                [--launcher {none,pytorch,slurm,mpi}]  
                [--local_rank LOCAL_RANK]  
                config  
train.py: error: the following arguments are required: config
```
=> python tools/train.py configs/pig/bk_yolox_onlybbox.py

## 최종 설정

최종설정사항은 다음의 화일에서 확인가능 
work_dir/{config}/{config}.py 

