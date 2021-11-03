import streamlit as st
import pandas as pd
import numpy as np
import nltk
from nltk.tag import CRFTagger
import pathlib
from docx import Document
import slate3k as slate
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
import gensim
from wordcloud import WordCloud
import matplotlib.pyplot as plt

st.set_option('deprecation.showPyplotGlobalUse', False)
def main():
    st.title("Aplikasi Text Preprocessing")
    st.markdown("Aplikasi ini digunakan untuk melakukan preprocessing text dari file berbahasa Indonesia")
    text = show_text()
    st.markdown("""---""")
    if(st.button("Preprocess")):
        #Tokenization
        TokenizedText = tokenization(text)
        st.header("TOKENIZATION")
        st.markdown(">Tokenization adalah proses untuk mengubah teks menjadi potongan token yang sudah di lowercase dan dihilangkan tanda bacanya")
        st.text_area(label='',height=200 , value=TokenizedText)
        st.markdown("""---""")

        #Stemming
        StemmedText = stemming(TokenizedText)
        st.header("STEMMING")
        st.markdown(">Stemming adalah proses untuk mengubah suatu kata menjadi kata dasarnya")
        st.text_area(label='',height=200 , value=StemmedText)
        st.markdown("""---""")

        #Stopword Removal
        CleanText = stopword_removal(StemmedText)
        st.header("STOPWORD REMOVAL")
        st.markdown(">Pada Proses Stopword Removal semua kata yang tidak memiliki makna akan dihilangkan")
        st.text_area(label='',height=200 , value=CleanText)
        st.markdown("""---""")

        #POS Tagging
        word_pos_count = pos_tagging(CleanText)
        st.header("POS TAGGING")
        st.markdown(">Berikut adalah hasil proses POS Tagging")
        st.table(word_pos_count)
        csv_file = convert_df(word_pos_count)
        st.download_button(
            label="Download CSV",
            data=csv_file,
            file_name='df_POS_Tagged.csv',
            mime='text/csv')
        #st.dataframe(word_pos_count, 2000, 500)
        st.markdown("""---""")

        #WordCloud
        wordCloud = word_cloud(CleanText)
        st.header("WORD CLOUD")
        st.markdown(">Berikut adalah gambar Word Cloud dari Text yang telah melalui preprocessing")
        plt.figure()
        plt.imshow(wordCloud, interpolation='bilinear')
        plt.axis("off")
        st.pyplot()

def show_text():
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        text = uploaded_file.read()
        text = text.decode("utf-8")
        st.text_area(label='',height=200 , value=text)
        return text

@st.cache
def tokenization(text):
    text = text.lower()
    LoweredText = word_tokenize(text)
    TokenizedText = [word for word in LoweredText if word.isalnum()]
    return TokenizedText

@st.cache
def stemming(TokenizedText):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    StemmedText = []
    for kata in TokenizedText:
        StemmedText.append(stemmer.stem(kata))
    return StemmedText

@st.cache
def stopword_removal(StemmedText):
    stop_words = stopwords.words('indonesian')
    CleanText = []
    for token in StemmedText:
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3 and token not in stop_words:
            CleanText.append(token)
    return CleanText

@st.cache
def pos_tagging(CleanText):
    ct = CRFTagger()
    ct.set_model_file('all_indo_man_tag_corpus_model.crf.tagger')
    PosTagged = ct.tag_sents([CleanText])
    WordBank = []
    for word in PosTagged[0]:
        WordBank.append(word)

    #JADIKAN DATAFRAME
    df = pd.DataFrame(WordBank, columns=['Word','POS'])
    counts = df['POS'].value_counts()

    #MENGHILANGKAN DATA BERULANG DAN MENAMBAHKAN FREKUENSI KEMUNCULAN
    word_count = pd.DataFrame(df.Word.value_counts().reset_index())
    word_count.columns = ['Word', 'Frequency']
    df.drop_duplicates("Word",inplace=True)
    word_pos_count = pd.merge(df,word_count,on='Word')

    #Add POS Description
    file = 'Penn-Treebank-POS-Tag.csv'
    treebank = pd.read_csv(file)
    treebank.drop(columns = 'Number', inplace=True)

    value = treebank["Description"].to_list()
    key = treebank['Tag'].to_list()
    pos_dict = {}
    values = treebank["Description"].to_list()
    keys = treebank['Tag'].to_list()
    for key in keys:
        for value in values:
            pos_dict[key] = value
            values.remove(value)
            break 
    keterangan = []
    for pos in word_pos_count['POS']:
        if pos in pos_dict.keys():
            keterangan.append(pos_dict[pos])
        else:
            keterangan.append("Tidak Diketahui")
    word_pos_count['Keterangan'] = keterangan
    word_pos_count.sort_values('Frequency',ascending=False,ignore_index=True,inplace=True)
    return word_pos_count

@st.cache
def convert_df(df):
    return df.to_csv().encode('utf-8')


@st.cache
def word_cloud(CleanText):
    cloud = " ".join(CleanText)
    wordCloud = WordCloud(width=800, height=400,collocations = False, background_color = 'white').generate(cloud)
    return wordCloud 

if __name__ == "__main__":
	main()