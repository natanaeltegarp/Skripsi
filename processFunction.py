from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import textstat as ts
import numpy as np

### INITIALIZATION ###
stem = StemmerFactory()
idn_stemmer = stem.create_stemmer()

### PREPROCESSING ###
def preprocess_text(text):
    text = str(text).lower()
    text = idn_stemmer.stem(text)
    return text

def apply_preprocess(kolom):
    kolom = kolom.apply(preprocess_text)
    return kolom

### FEATURE ###
def unigram_cosim_feature(kolom_kunci, kolom_jawab):
    vectorizer_tfidf_unigram = TfidfVectorizer(ngram_range=(1,1))
    tfidf_features_unigram = vectorizer_tfidf_unigram.fit_transform(kolom_jawab)
    tfidf_kunci_unigram = vectorizer_tfidf_unigram.transform(kolom_kunci)
    unigram_cosim = [
        cosine_similarity(tfidf_features_unigram[i], tfidf_kunci_unigram[i])[0, 0] if tfidf_features_unigram[i].nnz > 0 else 0
        for i in range(tfidf_features_unigram.shape[0])
    ]
    return unigram_cosim

def bigram_cosim_feature(kolom_kunci, kolom_jawab):
    vectorizer_tfidf_bigram = TfidfVectorizer(ngram_range=(2,2))
    tfidf_features_bigram = vectorizer_tfidf_bigram.fit_transform(kolom_jawab)
    tfidf_kunci_bigram = vectorizer_tfidf_bigram.transform(kolom_kunci)
    bigram_cosim = [
        cosine_similarity(tfidf_features_bigram[i], tfidf_kunci_bigram[i])[0, 0] if tfidf_features_bigram[i].nnz > 0 else 0
        for i in range(tfidf_features_bigram.shape[0])
    ]
    return bigram_cosim

def wcr_feature(kolom_kunci, kolom_jawab):
    wc_ratio = []
    jawaban_list = kolom_jawab.tolist()
    kunci_jawaban = kolom_kunci.tolist()[0]
    word_limit = ts.lexicon_count(kunci_jawaban)
    for jawaban_siswa in jawaban_list:
        word_count = ts.lexicon_count(jawaban_siswa)
        wc_ratio.append(1-(word_count/word_limit))
    return wc_ratio

def features_combine(*arrays):
    reshaped_array = [np.array(arr).reshape(-1, 1) for arr in arrays]
    return np.hstack(reshaped_array)