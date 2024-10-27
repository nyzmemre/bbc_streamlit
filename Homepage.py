import streamlit as st
import joblib

# nmf modelini yüklüyorum projeme.
nmf_model = joblib.load('nmf_model.pkl')


st.title('Category Analysis')
st.write("You can categorise news descriptions by analysing them.")
# Kullanıcıdan description alacağım.
user_input = st.text_area("Please enter the text you want to analyse here:")

tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

categories = [
   'Sports & World Cup',
   'UK Politics & Governance',
   'Premier League & Football Competitions',
   'News Commentary & Analysis',
   'Crime & Police Investigations',
   'Humanitarian Crises & Mass Casualties',
   'Russia-Ukraine Conflict & Invasion'
]

if st.button('Predict'):
    if user_input:
        # Kullanıcıdan alınan metni vektörleştir
        user_input_tfidf = tfidf_vectorizer.transform([user_input])
        
        # NMF modeli ile konu tahmini yap
        topic_distribution = nmf_model.transform(user_input_tfidf)

        # En yüksek olasılık değerini ve buna karşılık gelen kategoriyi bul
        max_index = topic_distribution[0].argmax()
        st.write(f"Estimated Category: {categories[max_index]}")
    else:
        st.warning('Please enter a text.')
