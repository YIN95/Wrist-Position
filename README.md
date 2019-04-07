# Wrist-Position

### Project Summary

### TODO
- [x] Pre-process video
- [x] Collect and label the data  
- [x] Load data for training
- [x] Build up the model
- [ ] Training the model
- [ ] Merge two model for adaptive learning
- [ ] Evaluate the compare the performance

### Library Versions

- Python v3.5.2
- numpy v1.15.4
- matplotlib v3.0.2
- cv2 V3.4.4
- tensorflow-gpu v1.12.0
- keras v2.2.4

### Data Set
The data file looks like below
```
DataSet/
    |->2018-12-28/
        |->image/
            |-> 0.jpg
            |-> 1.jpg
            |-> 2.jpg
            |-> 3.jpg
            ...
        |->label/
            |-> label.json
    |->2018-12-29/
        ...
```

### Usage
To collect new data, run the scripts ./preProcessing/preprocess.sh  
press c to collect the data, press q to exit.  
```
cd preProcessing
sh preprocess.sh
```

### Licence

MIT License

Note: If you find this project useful, please include reference link in your work.
