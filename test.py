from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import pickle

csv_file_path = 'testSet-4q-features.csv'
data_csv = pd.read_csv(csv_file_path)

with open("trainSet-8q-model.pkl", "rb") as file:
    model = pickle.load(file)

idsoal_list = data_csv['IDPSJ'].unique()

for idsoal in idsoal_list:
    print(f"Testing for IDSoal: {idsoal}")
    subset = data_csv[data_csv['IDPSJ'] == idsoal]

    x = subset[['Cosim-Unigram','Cosim-Bigram','TypeTokenRatio','WordCountRatio']]
    y = subset['label']

    y_pred = model.predict(x)
    cm_csv = confusion_matrix(y, y_pred)
    cm_report_csv = classification_report(y, y_pred)

    print(cm_csv)
    print(cm_report_csv)