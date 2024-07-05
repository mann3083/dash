apt-get update
apt-get install -y libasound2 libpulse0
streamlit run app.py --server.port 8000 --server.address 0.0.0.0