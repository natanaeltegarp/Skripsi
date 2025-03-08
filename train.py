from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import pandas as pd
import numpy as np
import processFunction as pf

### INITIALIZATION ###
# Indonesian Stemmer
stem = StemmerFactory()
idn_stemmer = stem.create_stemmer()

# Load dataset
csv_file_path = 'trainDataset.csv'
data_csv = pd.read_csv(csv_file_path)
######################

### PREPROCESSING ###
data_csv['answer'] = pf.apply_preprocess(data_csv['answer'])
data_csv['answerKeys'] = pf.apply_preprocess(data_csv['answerKeys'])
#####################

### TRAINING INITIALIZATION ###
model = SVC(kernel='rbf')

idsoal_list = data_csv['IDPSJ'].unique()

###############################

for idsoal in idsoal_list:
    print(f"Training for IDSoal: {idsoal}")
    subset = data_csv[data_csv['IDPSJ'] == idsoal]

    ### FEATURE EXTRACTION ###
    # FEATURE 1
    unigram_feature = pf.unigram_cosim_feature(subset['answerKeys'],subset['answer'])

    # FEATURE 2
    bigram_feature = pf.bigram_cosim_feature(subset['answerKeys'],subset['answer'])
    
    #FEATURE 3
    wc_ratio_feature = pf.wcr_feature(subset['answerKeys'],subset['answer'])
    
    # Combining features
    x_final = np.hstack((np.array(unigram_feature).reshape(-1,1), np.array(bigram_feature).reshape(-1,1), np.array(wc_ratio_feature).reshape(-1,1)))
    ##########################
    
    y = subset['labela']

    #Training
    model.fit(x_final, y)

print("Training completed")

# TESTING #
csv_file_path = 'testDataset.csv'
data_csv = pd.read_csv(csv_file_path)

data_csv['answer'] = pf.apply_preprocess(data_csv['answer'])
data_csv['answerKeys'] = pf.apply_preprocess(data_csv['answerKeys'])

idsoal_list = data_csv['IDPSJ'].unique()

for idsoal in idsoal_list:
    print(f"Testing for IDSoal: {idsoal}")
    subset = data_csv[data_csv['IDPSJ'] == idsoal]

    ### FEATURE EXTRACTION ###
    # FEATURE 1
    unigram_feature = pf.unigram_cosim_feature(subset['answerKeys'],subset['answer'])
    
    # FEATURE 2
    bigram_feature = pf.bigram_cosim_feature(subset['answerKeys'],subset['answer'])
    
    #FEATURE 2
    wc_ratio_feature = pf.wcr_feature(subset['answerKeys'],subset['answer'])
    
    # Combining features
    x_test = np.hstack((np.array(unigram_feature).reshape(-1,1), np.array(bigram_feature).reshape(-1,1), np.array(wc_ratio_feature).reshape(-1,1)))
    ##########################
    
    y_test = subset['labela']

    #Testing
    y_pred = model.predict(x_test)
    cm_csv = confusion_matrix(y_test, y_pred)
    cm_report_csv = classification_report(y_test, y_pred)

    print(cm_csv)
    print(cm_report_csv)