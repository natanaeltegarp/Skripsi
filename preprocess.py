import pandas as pd
import processFunction as pf

### INITIALIZATION ###

csv_file_path = 'stelma-test-split.csv'
data_csv = pd.read_csv(csv_file_path)
######################

### PREPROCESSING ###
data_csv['answer'] = pf.apply_preprocess(data_csv['answer'])
data_csv['answerKeys'] = pf.apply_preprocess(data_csv['answerKeys'])
#####################

### FEATURE EXTRACTION ###
# FEATURE 1
unigram_feature = pf.unigram_cosim_feature(data_csv['answerKeys'],data_csv['answer'])

# FEATURE 2
bigram_feature = pf.bigram_cosim_feature(data_csv['answerKeys'],data_csv['answer'])

#FEATURE 3
tt_ratio_feature = pf.ttr_feature(data_csv['answer'])

#FEATURE 4
wc_ratio_feature = pf.wcr_feature(data_csv['answerKeys'],data_csv['answer'])

# Combining features
x_final = pf.features_combine(data_csv['IDJwb'],data_csv['IDPSJ'],unigram_feature,bigram_feature,tt_ratio_feature,wc_ratio_feature,data_csv['labela'],data_csv['label'])
x_final.columns = ['IDJwb','IDPSJ','Cosim-Unigram','Cosim-Bigram','TypeTokenRatio','WordCountRatio','labela','label']
##########################

x_final.to_csv(csv_file_path[:-4]+'-features.csv',index=False)