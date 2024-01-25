import streamlit as st
import streamlit_antd_components as sac
import pandas as pd
import matplotlib.pyplot as plt
from assets import *

st.set_page_config(
    page_title="SVR",
    page_icon="computer",
    layout="centered",
    initial_sidebar_state="expanded",
)

if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'n' not in st.session_state:
    st.session_state.n = None

with st.sidebar:
    selected = sac.menu([
        sac.MenuItem('Home', icon="house"),
        sac.MenuItem('Dataset', icon="book"),
        sac.MenuItem('Normalisasi', icon="bar-chart-line"),
        sac.MenuItem('Split Data', icon="hourglass-split"),
        sac.MenuItem('SVR', icon="activity"),
    ], open_all=False)

if selected == "Home":
    st.title("Menu Home")

if selected == "Dataset":
    st.title("Menu Dataset")
    st.subheader("Dataset")
    data = dataset()
    st.dataframe(data)
    st.subheader("Grafik")
    idr_values = data["IDR"].values.reshape(-1, 1)
    plt.figure(figsize=(16, 8))
    plt.plot(data['Tanggal'], data['IDR'])
    plt.title('Grafik Dataset')
    plt.xlabel('Tanggal')
    plt.ylabel('IDR')
    plt.grid(True)
    st.pyplot(plt.gcf())

if selected == "Normalisasi":
    st.title("Menu Normalisasi Data")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Data Normal")
        data = dataset()
        st.dataframe(data)
    with col2:
        st.subheader("Data Normalisasi")
        data = dataset()
        idr_values = data["IDR"].values.reshape(-1, 1)
        st.session_state.scaler = MinMaxScaler(feature_range=(0, 1))
        idr_normalized = st.session_state.scaler.fit_transform(idr_values)
        data["IDR"] = idr_normalized
        st.dataframe(data)

if selected == "Split Data":
    st.title("Menu Split Data")
    st.subheader("Pilih Persentase Split Data")
    n = st.selectbox("",("70:30", "80:20", "90:10"), index=None, placeholder="Split Data 80:20",)
    if n == None:
        st.session_state.n = 8
    if n == "70:30":
        st.session_state.n = 7
    if n == "80:20":
        st.session_state.n = 8
    if n == "90:10":
        st.session_state.n = 9
    col1, col2 = st.columns(2)
    split = split_data(st.session_state.n)
    with col1:
        st.subheader("Panjang Data Training")
        st.subheader(len(split[0]))
    with col2:
        st.subheader("Panjang Data Testing")
        st.subheader(len(split[1]))
    st.session_state.data_train = split[0]
    st.session_state.data_test = split[1]
    plt.figure(figsize=(16, 8))
    plt.plot(st.session_state.data_train['Tanggal'], st.session_state.data_train['IDR'], color='red', label='Training Data')
    plt.plot(st.session_state.data_test['Tanggal'], st.session_state.data_test['IDR'], color='blue', label='Testing Data')
    plt.title('Ploting Data Training dan Testing')
    plt.xlabel('Tanggal')
    plt.ylabel('IDR')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt.gcf())

if selected == "SVR":
    st.title("Prediksi dengen SVR")
    split = split_data(st.session_state.n)
    st.subheader("Panjang Timestep")
    t = st.slider('', min_value=5, max_value=15, value=5, step=5)
    sac.divider(label='PARAMETER SVR', icon='ubuntu', align='center', color='gray')
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.subheader("C")
        c = st.slider('', min_value=5, max_value=20, value=10, step=5)
    with col2:
        st.subheader("Gamma")
        gamma = st.slider('', min_value=5, max_value=20, value=5, step=5)
    with col3:
        st.subheader("Epsilon")
        epsilon = st.slider('', min_value=0.1, max_value=1.0, value=0.1, step=0.2, format="%.1f")
    with col4:
        st.subheader("Degree")
        degree = st.slider('', min_value=3, max_value=12, value=3, step=3)
    st.subheader("Kernel")
    kernel = st.selectbox("",("linear", "poly", "rbf", "sigmoid"), index=None, placeholder="Kernel SVR",)
    if c == None:
        c = 10
    if gamma == None:
        gamma = 5
    if epsilon == None:
        epsilon = 0.1
    if degree == None:
        degree = 3
    if kernel == None:
        kernel = "poly"
    sac.divider(label='Keterangan Parameter', icon='ubuntu', align='center', color='gray')
    g1, g2, g3, g4, g5, g6 = st.columns(6)
    # g5, g6 = st.columns(2)
    with g1:
        st.write("Nilai C")
        st.write(c)
    with g2:
        st.write("Nilai Gamma")
        st.write(gamma)
    with g3:
        st.write("Nilai Epsilon")
        st.write(epsilon)
    with g4:
        st.write("Nilai Degree")
        st.write(degree)
    with g5:
        st.write("Tipe Kernel")
        st.write(kernel)
    with g6:
        st.write("Nilai Timestep")
        st.write(t)
    
    if st.button("Proses"):
        st.subheader("Nilai Mape")
        model = svr(st.session_state.n, t, c, gamma, epsilon, degree, kernel)
        st.subheader(f"Nilai Mape = {round(model[0],2)}")
        data_real = np.array(model[1])
        hasil_prediksi = np.array(model[2])
        plt.figure(figsize=(16, 8))
        plt.plot(data_real)
        plt.plot(hasil_prediksi)
        plt.title('Perbandingan Data Real dan Hasil Prediksi')
        plt.xlabel('Data ke-')
        plt.ylabel('Nilai Data')
        plt.legend(['Real', 'Prediksi'])
        plt.grid()
        st.pyplot(plt.gcf())
    
    