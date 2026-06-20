# 🪖 Helmet Detection

Aplikasi deteksi penggunaan helm pada gambar menggunakan **HOG (Histogram of Oriented Gradients)** sebagai feature extraction dan **SVM (Support Vector Machine)** sebagai classifier. Dataset di-balance menggunakan **SMOTE** untuk menangani imbalanced class.

## 📊 Hasil Model

| Metrik | Nilai |
|---|---|
| Accuracy | 78.4% |
| Precision | 79.7% |
| Recall | 90.6% |
| F1-score | 84.8% |

## 📁 Struktur Project

```
.
├── app.py                          # Streamlit app
├── model/
│   └── svm_helmet_model_final.pkl  # Model SVM yang sudah di-train
├── requirements.txt
└── README.md
```

## 🚀 Cara Menjalankan Lokal

```bash
# 1. Clone repo
git clone <repo-url>
cd <repo-name>

# 2. Install dependencies
pip install -r requirements.txt

# 3. Jalankan Streamlit
streamlit run app.py
```

## 🧠 Pipeline Model

1. **Preprocessing** — crop bounding box dari anotasi XML, resize ke 128×128
2. **Feature Extraction** — HOG (9 orientations, 8×8 pixels per cell, 2×2 cells per block)
3. **Balancing** — SMOTE (sampling_strategy=0.6)
4. **Model** — SVM dengan kernel RBF (C=10, gamma='scale'), hyperparameter di-tuning via GridSearchCV
5. **Evaluasi** — dibandingkan dengan Random Forest, KNN, dan Gradient Boosting

## 📦 Dataset

Dataset tidak disertakan di repo ini karena ukurannya besar. Gunakan dataset helmet detection dengan struktur anotasi XML (format Pascal VOC) berisi label `With Helmet` / `Without Helmet`.