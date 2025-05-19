from sklearn.metrics import confusion_matrix, classification_report, make_scorer, root_mean_squared_error
import pandas as pd
import numpy as np
import pickle

csv_file_path = 'stelma-test-split-features.csv'
data_csv = pd.read_csv(csv_file_path)

with open("stelma-train-split-KNN-model.pkl", "rb") as file:
    model = pickle.load(file)

model_name = "KNN"
idsoal_list = data_csv['IDPSJ'].unique()

rmse_list = []

for idsoal in idsoal_list:
    print(f"Testing for IDSoal: {idsoal}")
    subset = data_csv[data_csv['IDPSJ'] == idsoal]

    x = subset[['Cosim-Unigram','Cosim-Bigram','TypeTokenRatio','WordCountRatio']]
    y = subset['label']

    y_pred = model.predict(x)
    cm_csv = confusion_matrix(y, y_pred)
    cm_report_csv = classification_report(y, y_pred)
    rmse = root_mean_squared_error(y, y_pred)
    rmse_list.append(rmse)

    print(cm_csv)
    print(cm_report_csv)
    print(rmse)
rmse_mean = np.mean(rmse_list)
print(f"RMSE score:  {rmse_mean:.4f}")


# x = data_csv[['Cosim-Unigram','Cosim-Bigram','TypeTokenRatio','WordCountRatio']]
# y = data_csv['label']

# y_pred = model.predict(x)
# cm_csv = confusion_matrix(y, y_pred)
# cm_report_csv = classification_report(y, y_pred)
# rmse = root_mean_squared_error(y, y_pred)

# print(model_name)
# print(cm_csv)
# print(cm_report_csv)
# print(f"RMSE score:  {rmse:.4f}")