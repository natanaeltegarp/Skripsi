from sklearn.svm import SVC
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import root_mean_squared_error,make_scorer,precision_score
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
import processFunction as pf
import pickle

### INITIALIZATION ###

csv_file_path = 'trainSet-8q-features.csv'
data_csv = pd.read_csv(csv_file_path)

model = SVC(kernel='poly')
smote = SMOTE(random_state=42, k_neighbors=4)
prec_scorer = make_scorer(precision_score, average='macro', zero_division=1)
rmse_scorer = make_scorer(root_mean_squared_error)
idsoal_list = data_csv['IDPSJ'].unique()
#######################

for idsoal in idsoal_list:
    print(f"Training for IDSoal: {idsoal}")
    subset = data_csv[data_csv['IDPSJ'] == idsoal]

    x = subset[['Cosim-Unigram','Cosim-Bigram','TypeTokenRatio']]    
    # x = subset[['Cosim-Unigram','Cosim-Bigram','WordCountRatio']]
    y = subset['label']

    ### SMOTE ###
    x, y = smote.fit_resample(x,y)
    print("Data setelah proses SMOTE: ", np.bincount(y))
    #############

    ### VALIDATION ###
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    results_precision = cross_val_score(model, x, y, cv=kfold, scoring=prec_scorer)
    results_rmse = cross_val_score(model, x, y, cv=kfold, scoring=rmse_scorer)
    print("Precision: %.2f%%" % (results_precision.mean()*100.0))
    print("RMSE: %.4f" % results_rmse.mean())
    ##################

    #Visualization
    # pf.scatter_3d(x, y)

    #Training
    model.fit(x, y)

with open(csv_file_path[:-12]+"model.pkl","wb") as file:
    pickle.dump(model, file)
print("Training completed")