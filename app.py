import os
from flask import Flask, render_template, request, flash, redirect, url_for
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)
app.secret_key = 'svm-fraud-detection-2024-secret-key'

model = None
MODEL_PATH = 'svm_fraud_pipeline.pkl'

def load_model():
    global model
    try:
        model = joblib.load(MODEL_PATH)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None

load_model()

def get_plots_list():
    """Mengambil daftar file plot dari static/plots"""
    plots_dir = os.path.join('static', 'plots')
    plot_files = []
    
    if os.path.exists(plots_dir):
        for filename in sorted(os.listdir(plots_dir)):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.svg')):
                caption = filename.replace('_', ' ').replace('-', ' ').rsplit('.', 1)[0].title()
                plot_files.append({
                    'filename': filename,
                    'path': f'plots/{filename}',
                    'caption': caption
                })
    
    return plot_files

def validate_input(form_data):
    """Validasi input form"""
    errors = []
    
    # Validasi step
    try:
        step = int(form_data.get('step', ''))
        if step < 0:
            errors.append('Step harus >= 0')
    except (ValueError, TypeError):
        errors.append('Step harus berupa bilangan bulat')
    
    # Validasi type
    valid_types = ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN']
    type_val = form_data.get('type', '').strip().upper()
    if type_val not in valid_types:
        errors.append(f'Type harus salah satu dari: {", ".join(valid_types)}')
    
    # Validasi amount dan balance
    float_fields = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
    for field in float_fields:
        try:
            value = float(form_data.get(field, ''))
            if value < 0:
                errors.append(f'{field} harus >= 0')
        except (ValueError, TypeError):
            errors.append(f'{field} harus berupa bilangan')
    
    return errors

@app.route('/')
def home():
    """Halaman utama dengan form input dan daftar plot"""
    plots = get_plots_list()
    return render_template('home.html', plots=plots)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Route untuk prediksi fraud"""
    plots = get_plots_list()
    
    # Jika GET, tampilkan form (redirect ke home dengan scroll ke form)
    if request.method == 'GET':
        return render_template('home.html', plots=plots, scroll_to_form=True)
    
    # Jika POST, lakukan prediksi
    # Validasi input
    errors = validate_input(request.form)
    if errors:
        for error in errors:
            flash(error, 'error')
        return render_template('home.html', plots=plots, error=True)
    
    # Cek model
    if model is None:
        flash('Model belum dimuat. Silakan coba lagi.', 'error')
        return render_template('home.html', plots=plots, error=True)
    
    try:
        # Ambil input dari form
        input_data = {
            'step': int(request.form.get('step', 0)),
            'type': request.form.get('type', '').strip().upper(),
            'amount': float(request.form.get('amount', 0)),
            'oldbalanceOrg': float(request.form.get('oldbalanceOrg', 0)),
            'newbalanceOrig': float(request.form.get('newbalanceOrig', 0)),
            'oldbalanceDest': float(request.form.get('oldbalanceDest', 0)),
            'newbalanceDest': float(request.form.get('newbalanceDest', 0))
        }
        
        # Buat DataFrame 1 baris
        df_input = pd.DataFrame([input_data])
        
        # Prediksi
        prediction = model.predict(df_input)[0]
        
        # Ambil decision score untuk probabilitas
        decision_score = model.decision_function(df_input)[0]
        
        # Konversi decision score ke probabilitas (approximasi)
        # SVM tidak memberikan probabilitas langsung, jadi kita estimasi dari decision score
        # Score negatif = NORMAL, Score positif = FRAUD
        # Sigmoid untuk konversi score ke probabilitas
        prob_fraud = 1 / (1 + np.exp(-decision_score)) * 100
        prob_normal = 100 - prob_fraud
        
        # Tentukan label dan text
        pred_label = 'FRAUD' if prediction == 1 else 'NORMAL'
        confidence = prob_fraud if pred_label == 'FRAUD' else prob_normal
        
        if pred_label == 'FRAUD':
            pred_text = 'Transaksi terdeteksi sebagai FRAUD. Perlu verifikasi lebih lanjut sebelum melanjutkan transaksi.'
            reasons = [
                'Pola transaksi tidak sesuai dengan transaksi normal',
                'Kombinasi fitur menunjukkan karakteristik mencurigakan',
                'Model mendeteksi anomali dalam saldo dan jumlah transaksi'
            ]
        else:
            pred_text = 'Transaksi terdeteksi sebagai NORMAL. Transaksi aman untuk dilanjutkan.'
            reasons = [
                'Pola transaksi sesuai dengan transaksi normal',
                'Kombinasi fitur menunjukkan karakteristik yang wajar',
                'Tidak ada anomali yang terdeteksi dalam transaksi ini'
            ]
        
        return render_template(
            'home.html',
            plots=plots,
            pred_label=pred_label,
            pred_text=pred_text,
            prob_fraud=round(prob_fraud, 2),
            prob_normal=round(prob_normal, 2),
            confidence=round(confidence, 2),
            decision_score=round(decision_score, 4),
            reasons=reasons,
            input_values=input_data
        )
    
    except Exception as e:
        flash(f'Terjadi kesalahan saat prediksi: {str(e)}', 'error')
        return render_template('home.html', plots=plots, error=True)

@app.route('/visualizations')
def visualizations():
    """Halaman visualisasi"""
    plots = get_plots_list()
    return render_template('visualizations.html', plots=plots)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        message = request.form.get('message', '').strip()
        
        if not name or not email or not message:
            flash('Mohon lengkapi semua field.', 'error')
            return redirect(url_for('contact'))
        
        flash('Pesan Anda berhasil dikirim! Terima kasih atas feedback Anda.', 'success')
        return redirect(url_for('contact'))
    
    return render_template('contact.html')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
