# Fraud Detection SVM - Web Application

Aplikasi web untuk deteksi penipuan transaksi digital menggunakan **Support Vector Machine (SVM)** dengan akurasi ~95%.

## 🚀 Fitur

- **Prediksi Real-Time**: Input 7 fitur transaksi dan dapatkan prediksi FRAUD/NORMAL secara instan
- **Detail Probabilitas**: Menampilkan probabilitas NORMAL dan FRAUD dengan decision score
- **Visualisasi Data**: Gallery visualisasi dari analisis dataset dan performa model
- **Responsive Design**: Tampilan optimal di desktop dan mobile
- **Animasi Interaktif**: Animasi khusus untuk hasil prediksi

## 📋 Fitur Input

1. **Step**: Langkah dalam jam (integer >= 0)
2. **Type**: Jenis transaksi (PAYMENT, TRANSFER, CASH_OUT, DEBIT, CASH_IN)
3. **Amount**: Jumlah transaksi (float >= 0)
4. **Old Balance Origin**: Saldo awal pengirim
5. **New Balance Origin**: Saldo baru pengirim
6. **Old Balance Destination**: Saldo awal penerima
7. **New Balance Destination**: Saldo baru penerima

## 🛠️ Teknologi

- **Backend**: Flask (Python)
- **Machine Learning**: Scikit-learn (LinearSVC)
- **Frontend**: HTML5, CSS3, JavaScript
- **Deployment**: Railway (Gunicorn)

## 📦 Instalasi Lokal

1. Clone repository:
```bash
git clone https://github.com/imamrzkys/TUGAS-13-MECHINE-LEARNING.git
cd TUGAS-13-MECHINE-LEARNING
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Pastikan file model ada:
   - `svm_fraud_pipeline.pkl` (sudah termasuk di repository)

4. Generate visualisasi (opsional):
```bash
python generate_plots.py
```

5. Jalankan aplikasi:
```bash
python app.py
```

6. Buka browser: `http://127.0.0.1:5000/`

## 🚂 Deploy ke Railway

1. Login ke [Railway](https://railway.app)
2. Create New Project → Deploy from GitHub repo
3. Pilih repository: `TUGAS-13-MECHINE-LEARNING`
4. Railway akan otomatis detect:
   - `Procfile` untuk menjalankan Gunicorn
   - `requirements.txt` untuk install dependencies
5. Deploy akan otomatis berjalan

### Konfigurasi Railway

- **Build Command**: (otomatis)
- **Start Command**: `gunicorn app:app` (dari Procfile)
- **Port**: Railway akan set otomatis via `PORT` environment variable

## 📁 Struktur Project

```
TUGAS 13/
├── app.py                      # Flask application
├── generate_plots.py           # Script untuk generate visualisasi
├── svm_fraud_pipeline.pkl     # Model SVM yang sudah dilatih
├── requirements.txt            # Python dependencies
├── Procfile                    # Konfigurasi untuk Railway
├── .gitignore                  # Git ignore rules
├── static/
│   ├── css/
│   │   └── style.css          # Styling dengan tema Dark Navy + Cyan
│   ├── js/
│   │   └── app.js             # JavaScript untuk interaktivitas
│   ├── plots/                 # Folder visualisasi (12 plot PNG)
│   └── favicon.svg            # Favicon aplikasi
└── templates/
    ├── base.html              # Template base
    ├── home.html              # Halaman utama (form + hasil)
    ├── about.html             # Halaman tentang
    ├── contact.html           # Halaman kontak
    ├── visualizations.html    # Halaman visualisasi
    └── predict.html           # (Legacy, tidak digunakan)
```

## 🎨 Tema

- **Background**: Dark Navy (#0b1220)
- **Card**: #111a2e
- **Primary**: Cyan (#00e5ff)
- **Text**: #e6eefc
- **Badge Fraud**: #ff3b3b
- **Badge Normal**: #29d67d

## 📊 Model Information

- **Algoritma**: Linear Support Vector Classifier (LinearSVC)
- **Akurasi**: ~95%
- **Dataset**: Penipuan Transaksi Digital (6.3M+ transaksi)
- **Preprocessing**: StandardScaler + OneHotEncoder
- **Hyperparameter**: C=10 (tuned via GridSearchCV)

## 📝 Catatan

- File CSV dataset tidak di-push ke GitHub (terlalu besar, ~150MB+)
- Model `svm_fraud_pipeline.pkl` sudah termasuk (di-push)
- Visualisasi plot sudah di-generate dan di-push ke `static/plots/`
- Untuk generate ulang plot, jalankan `python generate_plots.py`

## 👤 Author

Imam Rizky Saputra

## 📄 License

This project is for educational purposes.

