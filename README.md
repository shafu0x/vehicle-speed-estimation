# vehicle-speed-estimation

If you want to know more about this project checkout my [medium post](https://medium.com/p/18b41babda4c/edit).

![video](vis/pred_label_vis.gif)


## Requirements
```Shell
pip3 install -r requirements.txt
```

## How to use

### Pre-trained model

You can simply use the model that I trained before. It is under `models`. Use the `inference.ipynb` to load the model and run an inference.

### Train yourself

You can train the network (EfficientNet) to predict the speed of a vehicle using optical flow. If you want to train yourself, you will need to create the optical flow images first and save them as .npy files in a directory of your choice. You can do this here: [SharifElfouly/opical-flow-estimation-with-RAFT](https://github.com/SharifElfouly/opical-flow-estimation-with-RAFT).


