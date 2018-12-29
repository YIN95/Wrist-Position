dataPath="../DataSet/2018-12-28/"
modelPath="./Model/"
modelName="model.h5"
dynamic=False
imgSize=64

python3 training.py --dataPath $dataPath --dynamic $dynamic --imgSize $imgSize --modelPath $modelPath --modelName $modelName