# Phone Price Regression from Scratch

A machine learning project implementing multiple linear regression from scratch using Gradient Descent to predict mobile phone prices based on various hardware and software features.

---

## 📁 Project Structure

```text
phone-price-regression-from-scratch/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/                  # original CSV, never edited
│   │   └── mobile_phone_price.csv
│   └── processed/            # cleaned output from your notebook
├── notebooks/
│   ├── 01-eda-and-cleaning.ipynb
│   └── 02-model-training-and-evaluation.ipynb
├── src/
│   ├── __init__.py
│   ├── preprocessing.py      # cleaning, train_test_split, scaler
│   ├── model.py              # linear_model (gradient descent), predict
│   └── metrics.py            # evaluate_model (MAE, MSE, RMSE, R²)
├── reports/
│   └── figures/              # all saved PNGs
└── tests/
    └── test_model.py
```

---

## 🚀 Getting Started

### 1. Prerequisites & Virtual Environment

Create and activate a virtual environment:

```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate  # On Linux / macOS
# .venv\Scripts\activate   # On Windows
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔬 Workflow

1. **Exploratory Data Analysis & Cleaning (`notebooks/01-eda-and-cleaning.ipynb`)**:
   - Inspect raw data in `data/raw/mobile_phone_price.csv`.
   - Handle missing values, outliers, and data types.
   - Export processed dataset to `data/processed/`.

2. **Core Modules (`src/`)**:
   - `preprocessing.py`: Feature scaling, train-test splitting, and pipeline helpers from scratch.
   - `model.py`: Linear regression implementation using Gradient Descent (vectorized with NumPy).
   - `metrics.py`: Custom evaluation metrics (MAE, MSE, RMSE, $R^2$).

3. **Model Training & Evaluation (`notebooks/02-model-training-and-evaluation.ipynb`)**:
   - Load processed data.
   - Train the scratch regression model with hyperparameter tuning (learning rate, iterations).
   - Evaluate performance and save figures to `reports/figures/`.

4. **Testing (`tests/`)**:
   - Run unit tests using `pytest`:
   ```bash
   pytest tests/
   ```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
