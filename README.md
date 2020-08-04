The Assignment of Visual Medial 2020(UTokyo)
===========
The file 'report.pdf' is my report.
-----------

## Prerequisites

- Python 3.6
- GPU Memory >= 6G
- Numpy
- Pytorch 0.4.0+

### Train
Train the model by
```bash
git clone 'Visual_Media_2020_assignment'
cd Visual_Media_2020_assignment
mkdir save_models
python train.py --gpu_ids 0 --batchsize 32  --data_path your_data_path
```
`--gpu_ids` which gpu to run.

`--data_path` the path of the training data.

`--batchsize` batch size.

I trained it on Market1501 trainset(on single P100 GPU/ 60 epoches):
![](https://github.com/ZelongZeng/Visual_Media_2020_assignment/blob/master/741596464475_.pic.jpg)

### Test
Test the model by
```bash
python test.py --gpu_ids 0 --batchsize 32  --test_path your_data_path
python evaluation.py
```
`--gpu_ids` which gpu to run.

`--test_path` the path of the testing data.
`--batchsize` batch size.

I tested it on Market1501 testest:
![](https://github.com/ZelongZeng/Visual_Media_2020_assignment/blob/master/751596474035_.pic.jpg)

