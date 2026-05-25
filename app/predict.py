import pickle 
import pandas as pd
import os
from sklearn.metrics import classification_report



def predict():
    with open("models/model.pkl",'rb') as files:
        model=pickle.load(files)

    x_test=pd.read_csv('after encoding/x_test.csv')
    y_test=pd.read_csv('splitting data/y_test.csv')

    testing=model.predict(x_test)

    mk_dir='test_matrics'
    os.makedirs(mk_dir,exist_ok=True)
    file_path=os.path.join(mk_dir,'test_metrecs.txt')
    report=classification_report(testing,y_test)
    with open(file_path,'w') as file:
        file.write(report)
        file.write('\n\n')
    print("success")


predict()

    
