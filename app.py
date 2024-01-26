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
        sac.MenuItem('Prediksi', icon="sliders"),
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
    if st.session_state.n is None:
        st.session_state.n = 8
    split = split_data(st.session_state.n)
    st.subheader("Panjang Timestep")
    t = st.slider('', min_value=4, max_value=6, value=4, step=2)
    sac.divider(label='PARAMETER SVR', icon='ubuntu', align='center', color='gray')
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("C")
        c = st.slider('', min_value=5, max_value=15, value=10, step=5)
    with col2:
        st.subheader("Gamma")
        gamma = st.slider('', min_value=5, max_value=15, value=5, step=5)
    with col3:
        st.subheader("Epsilon")
        epsilon = st.slider('', min_value=0.1, max_value=0.9, value=0.1, step=0.2, format="%.1f")   
    degree = 3
    kernel = "poly"
    if c == None:
        c = 10
    if gamma == None:
        gamma = 5
    if epsilon == None:
        epsilon = 0.1
    if degree == None:
        degree = 3
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
        st.subheader(f"Nilai Mape = {round(model[0],4)}")
        data_real = np.array(model[1])
        hasil_prediksi = np.array(model[2])
        chart_data = {"Data Real": data_real, "Hasil Prediksi": hasil_prediksi}
        plt.figure(figsize=(16, 8))
        plt.plot(data_real)
        plt.plot(hasil_prediksi)
        plt.title('Perbandingan Data Real dan Hasil Prediksi')
        plt.xlabel('Data ke-')
        plt.ylabel('Nilai Data')
        plt.legend(['Real', 'Prediksi'])
        plt.grid()
        st.pyplot(plt.gcf())
        # h1, h2 = st.columns(2)
        # with h1:
        #     st.subheader("Data Asli")
        #     st.dataframe(data_real)
        # with h2:
        #     st.subheader("Data Prediksi")
        #     st.dataframe(hasil_prediksi)

if selected == "Prediksi":
    st.title("Menu Prediksi Data")
    days = day()[0]
    besok = day()[1]
    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)
    input_data = {}
    with col1:
        input_data['num1'] = st.number_input(f"Masukkan Data Emas Tanggal {days[-4]}")
    with col2:
        input_data['num2'] = st.number_input(f"Masukkan Data Emas Tanggal {days[-3]}")
    with col3:
        input_data['num3'] = st.number_input(f"Masukkan Data Emas Tanggal {days[-2]}")
    with col4:
        input_data['num4'] = st.number_input(f"Masukkan Data Emas Tanggal {days[-1]}")
    
    if st.button("Prediksi"):
        temp_data = [input_data[f'num{i}'] for i in range(1, 5)]
        data = [temp_data]
        pred = prediksi(data)
        st.subheader(f"Hasil Prediksi Emas Tanggal {besok}")
        hasil = '{:,.0f}'.format(pred).replace(',', '.')
        st.subheader(f"Rp. {hasil}")