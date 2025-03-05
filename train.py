from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import textstat as ts
import pandas as pd
import numpy as np

### INITIALIZATION ###
# Indonesian Stemmer
stem = StemmerFactory()
idn_stemmer = stem.create_stemmer()

# Load dataset
csv_file_path = 'trainDataset.csv'
data_csv = pd.read_csv(csv_file_path)
######################

### PREPROCESSING ###
def preprocess_text(text):
    text = str(text).lower()
    text = idn_stemmer.stem(text)
    return text

data_csv['answer'] = data_csv['answer'].apply(preprocess_text)
data_csv['answerKeys'] = data_csv['answerKeys'].apply(preprocess_text)
#####################

### TRAINING INITIALIZATION ###
model = SVC(kernel='rbf')

idsoal_list = data_csv['IDPSJ'].unique()

x_train = data_csv['answer']
y_train = data_csv['labela']

vectorizer_tfidf_unigram = TfidfVectorizer(ngram_range=(1,1))
vectorizer_tfidf_bigram = TfidfVectorizer(ngram_range=(2,2))
###############################

for idsoal in idsoal_list:
    print(f"Training for IDSoal: {idsoal}")
    subset = data_csv[data_csv['IDPSJ'] == idsoal]

    ### FEATURE EXTRACTION ###
    # FEATURE 1
    # TF-IDF Unigram
    tfidf_features_unigram = vectorizer_tfidf_unigram.fit_transform(subset['answer'])
    tfidf_kunci_unigram = vectorizer_tfidf_unigram.transform(subset['answerKeys'])
    # Cosim Unigram
    unigram_cosim = [
        cosine_similarity(tfidf_features_unigram[i], tfidf_kunci_unigram[i])[0, 0] if tfidf_features_unigram[i].nnz > 0 else 0
        for i in range(tfidf_features_unigram.shape[0])
    ]

    # FEATURE 2
    # TF-IDF Bigram
    tfidf_features_bigram = vectorizer_tfidf_bigram.fit_transform(subset['answer'])
    tfidf_kunci_bigram = vectorizer_tfidf_bigram.transform(subset['answerKeys'])
    # Cosim Bigram
    bigram_cosim = [
        cosine_similarity(tfidf_features_bigram[i], tfidf_kunci_bigram[i])[0, 0] if tfidf_features_bigram[i].nnz > 0 else 0
        for i in range(tfidf_features_bigram.shape[0])
    ]

    #FEATURE 2
    # Word Count Ratio
    wc_ratio = []
    jawaban_list = subset['answer'].tolist()
    kunci_jawaban = subset['answerKeys'].tolist()[0]
    word_limit = ts.lexicon_count(kunci_jawaban)
    for jawaban_siswa in jawaban_list:
        word_count = ts.lexicon_count(jawaban_siswa)
        wc_ratio.append(1-(word_count/word_limit))

    # Combining features
    x_final = np.hstack((np.array(unigram_cosim).reshape(-1,1), np.array(bigram_cosim).reshape(-1,1), np.array(wc_ratio).reshape(-1,1)))
    ##########################
    
    y = subset['labela']


    #Training
    model.fit(x_final, y)

print("Training completed")

# TESTING #
csv_file_path = 'testDataset.csv'
data_csv = pd.read_csv(csv_file_path)

data_csv['answer'] = data_csv['answer'].apply(preprocess_text)
data_csv['answerKeys'] = data_csv['answerKeys'].apply(preprocess_text)

idsoal_list = data_csv['IDPSJ'].unique()

x_test = data_csv['answer']
y_test = data_csv['labela']

for idsoal in idsoal_list:
    print(f"Testing for IDSoal: {idsoal}")
    subset = data_csv[data_csv['IDPSJ'] == idsoal]

    ### FEATURE EXTRACTION ###
    # FEATURE 1
    # TF-IDF Unigram
    tfidf_features_unigram = vectorizer_tfidf_unigram.fit_transform(subset['answer'])
    tfidf_kunci_unigram = vectorizer_tfidf_unigram.transform(subset['answerKeys'])
    # Cosim Unigram
    unigram_cosim = [
        cosine_similarity(tfidf_features_unigram[i], tfidf_kunci_unigram[i])[0, 0] if tfidf_features_unigram[i].nnz > 0 else 0
        for i in range(tfidf_features_unigram.shape[0])
    ]

    # FEATURE 2
    # TF-IDF Bigram
    tfidf_features_bigram = vectorizer_tfidf_bigram.fit_transform(subset['answer'])
    tfidf_kunci_bigram = vectorizer_tfidf_bigram.transform(subset['answerKeys'])
    # Cosim Bigram
    bigram_cosim = [
        cosine_similarity(tfidf_features_bigram[i], tfidf_kunci_bigram[i])[0, 0] if tfidf_features_bigram[i].nnz > 0 else 0
        for i in range(tfidf_features_bigram.shape[0])
    ]

    #FEATURE 2
    # Word Count Ratio
    wc_ratio = []
    jawaban_list = subset['answer'].tolist()
    kunci_jawaban = subset['answerKeys'].tolist()[0]
    word_limit = ts.lexicon_count(kunci_jawaban)
    for jawaban_siswa in jawaban_list:
        word_count = ts.lexicon_count(jawaban_siswa)
        wc_ratio.append(1-(word_count/word_limit))

    # Combining features
    x_test = np.hstack((np.array(unigram_cosim).reshape(-1,1), np.array(bigram_cosim).reshape(-1,1), np.array(wc_ratio).reshape(-1,1)))
    ##########################
    
    y_test = subset['labela']


    #Testing
    y_pred = model.predict(x_test)
    cm_csv = confusion_matrix(y_test, y_pred)
    cm_report_csv = classification_report(y_test, y_pred)

    print(cm_csv)
    print(cm_report_csv)