import streamlit as st
from InsuranceAssistant import InsuranceAssistant

assistant = InsuranceAssistant()

st.title("Insurance Assistant")

if st.button('Recognize Speech'):
    text = assistant.recognize_from_microphone()
    st.write(text)