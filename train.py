from sklearn.svm import SVC
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import root_mean_squared_error,make_scorer,precision_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import pandas as pd
import numpy as np
import processFunction as pf
import pickle

### INITIALIZATION ###

csv_file_path = 'stelma-train-split-features.csv'
data_csv = pd.read_csv(csv_file_path)


model_name = "MSVM"
# model = SVC(kernel='linear')
# model = SVC(kernel='poly', degree=5)
model = SVC(kernel='rbf', gamma=100)
# model = LogisticRegression()
# model = KNeighborsClassifier(n_neighbors=3)
smote = SMOTE(random_state=42)

prec_scorer = make_scorer(precision_score, average='macro', zero_division=1)
precision_list = []
idsoal_list = data_csv['IDPSJ'].unique()
#######################

print(model_name)
# print("UC-BC-TTR")
# print("UC-BC-WCR")
# print("UC-TTR-WCR")
# print("BC-TTR-WCR")
print("UC-BC-TTR-WCR")
for idsoal in idsoal_list:
    print(f"Training for IDSoal: {idsoal}")
    subset = data_csv[data_csv['IDPSJ'] == idsoal]

    # x = subset[['Cosim-Unigram','Cosim-Bigram','TypeTokenRatio']]    
    # x = subset[['Cosim-Unigram','Cosim-Bigram','WordCountRatio']]
    # x = subset[['Cosim-Unigram','TypeTokenRatio','WordCountRatio']]
    # x = subset[['Cosim-Bigram','TypeTokenRatio','WordCountRatio']]
    x = subset[['Cosim-Unigram','Cosim-Bigram','TypeTokenRatio','WordCountRatio']]
    y = subset['label']
    
    #Visualization
    # pf.scatter_3d(x, y)

    ### SMOTE ###
    x, y = smote.fit_resample(x,y)
    print("Data setelah proses SMOTE: ", np.bincount(y))
    #############

    #Training
    model.fit(x, y)

    ### VALIDATION ###
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    results_precision = cross_val_score(model, x, y, cv=kfold, scoring=prec_scorer)
    precision_list.append(results_precision.mean())
    print(f"Precision: {results_precision.mean()*100.0:.2f}%")
    ##################

precision_list = pd.Series(precision_list)
precision_mean = precision_list.mean()
print(f"Precision score for this model: {precision_mean*100.0:.2f}%")

with open(csv_file_path[:-12]+model_name+"-model.pkl","wb") as file:
    pickle.dump(model, file)
print("Training completed")