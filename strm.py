import streamlit as st


with st.container(border=True):
    st.markdown("This is the 1st section")

with st.container(border=True):
    st.markdown("This is the 2nd section")

with st.container(border=True):
    st.markdown("This is the 3rd section")


 #python -m streamlit run strm.py --server.port 8000 --server.address 0.0.0.0  --server.enableCORS=false --server.enableXsrfProtection=false
 #python -m streamlit run strm.py --server.port 8000 --server.address 0.0.0.0 --server.enableCORS=false --server.enableXsrfProtection=false
 #python -m streamlit run strm.py --server.port 8000 --server.address 0.0.0.0 --server.enableWebsocketCompression false