# Fake Banknote Authentication

This project uses machine learning to detect fake banknotes based on image wavelet transform features (variance, skewness, curtosis, and entropy).

## Project Structure

- `data/`: Contains the `BankNote_Authentication.csv` dataset.
- `models/`: Contains the trained `model.pkl`.
- `notebooks/`: Contains the research and training notebook (`banknote_analysis.ipynb`).
- `src/`: Contains the application source code.
  - `main.py`: FastAPI backend.
  - `static/`: Frontend assets (CSS, JS).
  - `templates/`: HTML templates.
- `requirements.txt`: Python dependencies.
- `.gitignore`: Files to be ignored by Git.

## How to Run

1.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Start the server**:
    ```bash
    python src/main.py
    ```
    Or:
    ```bash
    uvicorn src.main:app --reload
    ```

3.  **Access the application**:
    Open your browser and navigate to `http://localhost:8000`.

## Model Features

The model uses the following features for prediction:
- **Variance**: Variance of Wavelet Transformed image.
- **Skewness**: Skewness of Wavelet Transformed image.
- **Curtosis**: Curtosis of Wavelet Transformed image.
- **Entropy**: Entropy of image.
