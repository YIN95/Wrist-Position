import json


path = '/media/ywj/FIle/M-MyCode/Wrist-Position/preProcessing/DataSet/2018-12-28/label/label.json'

with open(path,'r') as load_f:
    load_dict = json.load(load_f)
    print(load_dict)

print(type(load_dict))
print(load_dict['3'])