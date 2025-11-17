import pandas as pd
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression

DATA_PATH = 'titanic_preprocessing.csv' 

# mengaktifkan autolog
mlflow.sklearn.autolog() 

# mengatur nama eksperimen di MLflow
# nama = "Modelling"
# mlflow.set_experiment(nama)

def run_modelling():
    print("Memulai proses modelling...")
    
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"Error: File tidak ditemukan di {DATA_PATH}")
        print("Pastikan path sudah benar")
        return
    
    # Pisahkan Fitur (X) dan Target (y)
    X = df.drop('Survived', axis=1)
    y = df['Survived']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Data siap. Memulai pelatihan model Logistic Regression...")

    # --- RUN LOGISTIC REGRESSION ---
    with mlflow.start_run():
        print("\nMelatih LogisticRegression...")
        model_lr = LogisticRegression(solver='liblinear', random_state=42)
        model_lr.fit(X_train, y_train)
        print("LogisticRegression selesai.")

    print("\nModel selesai dilatih.")


if __name__ == "__main__":
    run_modelling()