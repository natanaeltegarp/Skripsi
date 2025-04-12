from sklearn.svm import SVC
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import root_mean_squared_error,make_scorer,precision_score
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
import processFunction as pf
import pickle

### INITIALIZATION ###

csv_file_path = 'trainSet-8q-features.csv'
data_csv = pd.read_csv(csv_file_path)

model = SVC(kernel='rbf', gamma=100)
smote = SMOTE(random_state=42, k_neighbors=4)
prec_scorer = make_scorer(precision_score, average='macro', zero_division=1)
rmse_scorer = make_scorer(root_mean_squared_error)
precision_list = []
rmse_list = []
idsoal_list = data_csv['IDPSJ'].unique()
#######################

for idsoal in idsoal_list:
    print(f"Training for IDSoal: {idsoal}")
    subset = data_csv[data_csv['IDPSJ'] == idsoal]

    # x = subset[['Cosim-Unigram','Cosim-Bigram','TypeTokenRatio']]    
    # x = subset[['Cosim-Unigram','Cosim-Bigram','WordCountRatio']]
    # x = subset[['Cosim-Unigram','TypeTokenRatio','WordCountRatio']]
    # x = subset[['Cosim-Bigram','TypeTokenRatio','WordCountRatio']]
    x = subset[['Cosim-Unigram','Cosim-Bigram','TypeTokenRatio','WordCountRatio']]
    y = subset['label']

    ### NORMALIZATION ###
    # scaler = MinMaxScaler()
    # x = scaler.fit_transform(x)
    #####################
    
    ### SMOTE ###
    x, y = smote.fit_resample(x,y)
    print("Data setelah proses SMOTE: ", np.bincount(y))
    #############

    ### VALIDATION ###
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    results_precision = cross_val_score(model, x, y, cv=kfold, scoring=prec_scorer)
    results_rmse = cross_val_score(model, x, y, cv=kfold, scoring=rmse_scorer)
    precision_list.append(results_precision.mean())
    rmse_list.append(results_rmse.mean())
    print(f"Precision: {results_precision.mean()*100.0:.2f}%")
    print(f"RMSE: {results_rmse.mean():.4f}")
    ##################

    #Visualization
    # pf.scatter_3d(x, y)

    #Training
    model.fit(x, y)

precision_mean = np.mean(precision_list)
rmse_mean = np.mean(rmse_list)
print(f"Precision score for this model: {precision_mean*100.0:.2f}%")
print(f"RMSE score for this model: {rmse_mean:.4f}")

with open(csv_file_path[:-12]+"model.pkl","wb") as file:
    pickle.dump(model, file)
print("Training completed")