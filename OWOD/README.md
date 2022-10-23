## Continual Learning for Object Detection

#### Highlights

- Recent works suggest improving open world capability can improve continual learning
- Towards this attempt, the unknown identifiability of the Faster-RCNN object detector is explored
- Different post-processing techniques such as Soft-NMS and DIoU-NMS are studied for unknown identifiability


#### Installation
Please refer [INSTALL.md](INSTALL.md).

#### Quick Start

Data split and trained models: [[Google Drive Link 1]](https://drive.google.com/drive/folders/1Sr4_q0_m2f2SefoebB25Ix3N1VIAua0w?usp=sharing) [[Google Drive Link 2]](https://drive.google.com/drive/folders/11bJRdZqdtzIxBDkxrx2Jc3AhirqkO0YV?usp=sharing)

All config files can be found in: `configs/OWOD`

Sample command on a 4 GPU machine:
```python
python tools/train_net.py --num-gpus 4 --config-file <Change to the appropriate config file> SOLVER.IMS_PER_BATCH 4 SOLVER.BASE_LR 0.005
```

Kindly run `replicate.sh` to replicate results from the models shared on the Google Drive. 

Kindly check `run.sh` file for a task workflow.


#### Acknowledgement

Code is built upon [OWOD](github.com/josephKJ/OWOD). Many thanks to the authors for their implementation.
