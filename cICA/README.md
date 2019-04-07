# cICA-Extracting-Correlated-Signals

## Note: This project is public.

### Project Summary

### TODO
- [x] To understand the scripts
- [x] Re-write the code
- [ ] Collect new dataset
- [ ] Test on new dataset
- [ ] Summary

### Library Versions

- scipy v1.1.0 (necessary, if you use v1.0.1, you cannot import find_peaks)
- Python v3.6.5 (not testing on other version, but mostly 3.5+ will works)
- numpy v1.13.3
- matplotlib v2.2.2
- imageio v2.4.1
- sklearn v0.19.1

### Data Set
The data file looks like below
```
root/
    |->Data/
        |->DataSetName/
            |-> imageName1.png
            |-> imageName2.png
            |-> imageName3.png
            |-> imageName4.png
            ...
```

### Usage
To collect new data, press Y to collect, press N to stop, press Q to quit.  
press W to move up, press S to move down, press A to move left, press D to move right.
```
python getData_util.py
```

To extract the correlated signals
```
python cICA_Extractor.py
```


### References

[1] Wei Lu, Jagath C. Rajapakse: ICA with reference. ICA 2001  
[2] Zhi-Lin Zhang, Morphologically Constrained ICA for Extracting Weak Temporally Correlated Signals, Neurocomputing 71(7-9) (2008) 1669-1679

### Licence

MIT License

Note: If you find this project useful, please include reference link in your work.
