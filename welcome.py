import streamlit as st
import joblib

# nmf modelini yüklüyorum projeme.
nmf_model = joblib.load('nmf_model.pkl')

st.title('Kategori Analizi')
st.write('Haber açıklamalarını analiz ederek kategorilere ayırabilirsiniz.')

# Kullanıcıdan description alacağım.
user_input = st.text_area('Lütfen analiz etmek istediğiniz metni buraya girin:')

tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

categories = [
    "War and International Politics",
    "Sport News",
    "History and Past Events",
    "Career and Business Life",
    "Sport News",
    "Society and Security",
    "United Kingdom Policy"
]

if st.button('Tahmin Et'):
    if user_input:
        # Kullanıcıdan alınan metni vektörleştir
        user_input_tfidf = tfidf_vectorizer.transform([user_input])
        
        # NMF modeli ile konu tahmini yap
        topic_distribution = nmf_model.transform(user_input_tfidf)

        # En yüksek olasılık değerini ve buna karşılık gelen kategoriyi bul
        max_index = topic_distribution[0].argmax()
        st.write(f"Tahmin Edilen Kategori: {categories[max_index]}")
    else:
        st.warning('Lütfen bir metin girin.')
