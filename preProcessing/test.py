import pickle
path = '/media/ywj/FIle/M-MyCode/Wrist-Position/preProcessing/DataSet/2018-12-28/label/' 
fileTrain = path + 'label.pkl'
with open(fileTrain, 'rb') as f:
    data = pickle.load(f)
print(data)