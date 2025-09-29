# ⚕️ Obesity Prediction Using Machine Learning

This project predicts **obesity levels** based on lifestyle, eating habits, and demographic data.
It uses **LightGBM (LGBMClassifier)** with **scikit-learn pipelines**, including preprocessing and hyperparameter tuning.

---

## 📂 Dataset Description

The dataset (`ObesityDataSet2.csv`) contains demographic, lifestyle, and nutrition-related features.

### Features:

* **Gender**: Laki-laki / Wanita
* **Age**: Usia (tahun)
* **Height**: Tinggi (meter)
* **Weight**: Berat (kg)
* **family_history_with_overweight**: Riwayat keluarga dengan obesitas (ya/tidak)
* **FAVC**: Konsumsi makanan berkalori tinggi (ya/tidak)
* **FCVC**: Konsumsi sayuran (skala 1–3)
* **NCP**: Jumlah makanan utama per hari
* **CAEC**: Frekuensi ngemil (Tidak pernah / Kadang-kadang / Sering / Selalu)
* **SMOKE**: Status merokok (ya/tidak)
* **CH2O**: Asupan air (skala 1–3)
* **SCC**: Pemantauan kalori (ya/tidak)
* **FAF**: Aktivitas fisik (skala 0–3)
* **TUE**: Waktu penggunaan teknologi (skala 0–3)
* **CALC**: Konsumsi alkohol (Tidak pernah / Kadang-kadang / Sering / Selalu)
* **MTRANS**: Moda transportasi (Mobil / Motor / Sepeda / Transportasi Umum / Jalan kaki)
* **Target (NObeyesedad)**:

  * Insufficient Weight
  * Normal Weight
  * Overweight Level I
  * Overweight Level II
  * Obesity Type I
  * Obesity Type II
  * Obesity Type III

---

## 🛠️ Workflow

### 1. **Exploratory Data Analysis (EDA)**

* Checked missing values, duplicates, and datatypes.
* Fixed `Age` column (string → numeric).
* Removed duplicates and cleaned missing values.
* Visualized categorical features (pie + bar plots).
* Visualized numerical features (histograms + boxplots).
* Correlation analysis between numeric features.

### 2. **Preprocessing Pipeline**

* Numeric: imputation (median) + standard scaling.
* Binary categorical: imputation (most frequent) + one-hot encoding.
* Multiclass categorical: imputation (most frequent) + one-hot encoding (drop first).

### 3. **Model Comparison**

Models evaluated with **cross-validation (StratifiedKFold)**:

* RandomForestClassifier
* ExtraTreesClassifier
* DecisionTreeClassifier
* LightGBMClassifier (best)

### 4. **Hyperparameter Tuning**

Performed **GridSearchCV** with F1-macro scoring.
Best parameters were used to train the final **LGBMClassifier**.

### 5. **Model Saving**

Saved the tuned pipeline with preprocessing + model:

```bash
lgbm_best_pipeline.pkl
```

---

## 📊 Results

| Model                  | Mean Accuracy | F1 Macro |
| ---------------------- | ------------- | -------- |
| RandomForestClassifier | ~0.xx         | ~0.xx    |
| ExtraTreesClassifier   | ~0.xx         | ~0.xx    |
| DecisionTreeClassifier | ~0.xx         | ~0.xx    |
| **LGBMClassifier**     | **Best**      | **Best** |

➡️ **LGBMClassifier outperformed other models** due to:

* Gradient boosting focus on misclassified samples.
* Leaf-wise tree growth for deeper insights.
* Efficient handling of mixed numerical + categorical data.

---

## 🚀 Deployment

The saved pipeline (`.pkl`) can be loaded directly in:

* **Streamlit app** for interactive prediction.
* **FastAPI/Flask** for backend API.

Example usage:

```python
import joblib
import pandas as pd

# Load pipeline
model = joblib.load("lgbm_best_pipeline.pkl")

# Example input (dictionary)
sample = pd.DataFrame([{
    "Gender": "Male",
    "Age": 22,
    "Height": 1.75,
    "Weight": 70,
    "family_history_with_overweight": "yes",
    "FAVC": "no",
    "FCVC": 2,
    "NCP": 3,
    "CAEC": "Sometimes",
    "SMOKE": "no",
    "CH2O": 2,
    "SCC": "no",
    "FAF": 2,
    "TUE": 1,
    "CALC": "no",
    "MTRANS": "Public_Transportation"
}])

# Prediction
print(model.predict(sample))
```

---

## 📌 Future Improvements

* Deploy with **Streamlit Cloud** for live demo.
* Add SHAP / LIME explainability for feature importance.
* Expand dataset with more demographics.

---

## 📑 Citation

If you use this project, please cite:
**Gervasius Russell (2025), BINUS University – Obesity Prediction with LightGBM**

---
