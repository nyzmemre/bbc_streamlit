import streamlit as st

# About Us başlığı
st.title("About Us")

# Kişilerin bilgilerini temsil eden örnek liste
team_members = [
    {
        "name": "Gizem Özmen",
        "role": "Industrial Engineer (BSc) | Data Scientist",
        "bio": "Gizem is an expert in Quality Management Systems and Lean Manufacturing within the Automotive and FMCG sectors. She is highly motivated to continuously develop herself, with a particular focus on Data Science and Machine Learning.",
        "linkedin": "https://www.linkedin.com/in/gizem-ozmen-b5458637/",
        "image": "images/gizem.png"
    },
    {
        "name": "Melike Sevinç",
        "role": "Data Scientist | Instructor",
        "bio": "Melike wrote articles on cloud computing, and conducted interviews for bulutservisi.com. She taught information technology and robotics coding, and is now focused on data science education and research.",
        "linkedin": "https://www.linkedin.com/in/melikesevincms/",
        "image": "images/melike.jpeg"
    },
    
    {
        "name": "Mehmet Emre ÖZ",
        "role": "Instructor | Software Developer | Data Scientist",
        "bio": "Emre works in the field of education. He develops mobile applications and is interested in data science and machine learning.",
        "linkedin": "https://www.linkedin.com/in/nyzmemre/",
        "image": "images/emre.jpeg"
    }
]

# Her bir kişi için bilgilerin gösterilmesi
for member in team_members:
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.image(member["image"], width=150)
    
    with col2:
        st.markdown(f"### {member['name']}")
        st.markdown(f"**Role:** {member['role']}")
        st.markdown(f"{member['bio']}")
        st.markdown(f"[LinkedIn Profile]({member['linkedin']})")

    st.markdown("---")  # Ayırıcı çizgi
