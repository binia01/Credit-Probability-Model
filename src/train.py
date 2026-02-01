import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
import mlflow
import mlflow.sklearn
from src.data_processing import run_pipeline, create_preprocessing_pipeline, create_ml_pipeline

# Setup Directories
os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)

def load_data(file_path):
    """
    Loads the processed model-ready data.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Data loaded. Shape: {df.shape}")
        return df
    
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    
def eval_metrics(actual, pred, pred_proba):
    """Calculates comprehensive classification metrics."""
    return {
        "accuracy": accuracy_score(actual, pred),
        "precision": precision_score(actual, pred),
        "recall": recall_score(actual, pred),
        "f1": f1_score(actual, pred),
        "roc_auc": roc_auc_score(actual, pred_proba)
    }

def plot_confusion_matrix(y_test, y_pred, model_name):
    """Plots and saves confusion matrix."""
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    path = f"plots/cm-{model_name}.png"
    plt.savefig(path)
    plt.close()
    return path

def train_model(model_name, model_obj, param_grid, X_train, X_test, y_train, y_test):
    """Generic function to train, tune and log any model."""
    with mlflow.start_run(run_name=f"Train_{model_name}"):
        print(f"Training {model_name}...")

        # Hyperparameter Tuning (Grid Search)
        print(f"Running Grid Search for {model_name}...")
        grid = GridSearchCV(model_obj, param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_
        print(f"Best Params: {grid.best_params_}")
        
        # 2. Predictions
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]
        
        # 3. Metrics
        metrics = eval_metrics(y_test, y_pred, y_pred_proba)
        print(f"Performance: AUC={metrics['roc_auc']:.4f}, Accuracy={metrics['accuracy']:.4f}")
        
        # 4. Logging to MLflow
        mlflow.log_params(grid.best_params_)
        mlflow.log_metrics(metrics)
        
        # Log Artifacts (Confusion Matrix)
        cm_path = plot_confusion_matrix(y_test, y_pred, model_name)
        mlflow.log_artifact(cm_path)
        
        # 5. Log Model and Register
        mlflow.sklearn.log_model(best_model, artifact_path="model", registered_model_name=f"CreditRisk_{model_name}")
        
        # Save locally
        pickle.dump(best_model, open(f"models/{model_name.lower()}.pkl", "wb"))
        
        return metrics['roc_auc']

def save_preprocessing_pipeline():
    """
    Generates and saves the preprocessing pipeline needed for inference.
    This pipeline must be applied to raw customer data before model prediction.
    """
    try:
        print("Generating and saving preprocessing pipeline...")
        preprocessing_pipeline = create_preprocessing_pipeline()
        pickle.dump(preprocessing_pipeline, open("models/preprocessing_pipeline.pkl", "wb"))
        print("✅ Preprocessing pipeline saved to models/preprocessing_pipeline.pkl")
        return preprocessing_pipeline
    except Exception as e:
        print(f"Warning: Could not save preprocessing pipeline: {e}")
        return None

def save_ml_pipeline(ml_pipeline):
    """
    Saves the ML transformation pipeline (WoE + StandardScaler).
    This must be applied after preprocessing.
    """
    try:
        print("Saving ML pipeline (WoE + StandardScaler)...")
        pickle.dump(ml_pipeline, open("models/ml_pipeline.pkl", "wb"))
        print("✅ ML pipeline saved to models/ml_pipeline.pkl")
    except Exception as e:
        print(f"Warning: Could not save ML pipeline: {e}")

def save_rfm_scaler_from_raw():
    """
    Creates and saves a StandardScaler for RFM features using RAW aggregated data.
    This is critical for the dashboard to work with unscaled RFM inputs.
    
    The model_ready_data.csv has already-scaled RFM features, so we need to:
    1. Load raw data
    2. Run aggregation (without WoE/StandardScaler)
    3. Fit scaler on the unscaled RFM values
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from src.data_processing import TimeFeatureExtractor, CategoricalEncoder, CustomerAggregator, MissingValueHandler, RiskProxyLabeler
    
    try:
        print("Creating RFM scaler from RAW data (not pre-scaled)...")
        
        # Load raw data
        raw_df = pd.read_csv("./data/raw/data.csv")
        print(f"Loaded raw data: {raw_df.shape}")
        
        # Run preprocessing up to aggregation (no WoE/StandardScaler)
        preprocessing_pipeline = Pipeline([
            ('time_extraction', TimeFeatureExtractor()),
            ('categorical_encoding', CategoricalEncoder(columns=['ChannelId', 'ProductCategory', 'PricingStrategy'])),
            ('aggregation', CustomerAggregator()),
            ('missing_imputation', MissingValueHandler(strategy='mean')),
        ])
        
        aggregated_df = preprocessing_pipeline.fit_transform(raw_df)
        print(f"Aggregated data shape: {aggregated_df.shape}")
        
        rfm_features = ['Recency', 'Frequency', 'Monetary_Total', 'Monetary_Mean', 'Monetary_Std']
        
        # Check if all RFM features exist
        available_rfm = [col for col in rfm_features if col in aggregated_df.columns]
        if len(available_rfm) < 5:
            print(f"Warning: Only found {len(available_rfm)} RFM features: {available_rfm}")
            return None, None
        
        # Generate risk labels using RiskProxyLabeler
        print("Generating risk labels for RFM model training...")
        labeler = RiskProxyLabeler(n_clusters=3, random_state=42)
        labeler.fit(aggregated_df)
        labeled_df = labeler.transform(aggregated_df)
        
        # Fit scaler on UNSCALED RFM values
        rfm_scaler = StandardScaler()
        rfm_scaler.fit(aggregated_df[rfm_features])
        
        # Print stats for debugging - these should show real data ranges
        print(f"RFM Scaler stats (from raw aggregated data):")
        for i, feat in enumerate(rfm_features):
            print(f"  {feat}: mean={rfm_scaler.mean_[i]:.2f}, std={rfm_scaler.scale_[i]:.2f}")
        
        pickle.dump(rfm_scaler, open("models/rfm_scaler.pkl", "wb"))
        print("✅ RFM scaler saved to models/rfm_scaler.pkl")
        
        # Return both scaler and labeled data for RFM model training
        return rfm_scaler, labeled_df, rfm_features
    except Exception as e:
        print(f"Warning: Could not save RFM scaler: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def train_rfm_model(labeled_df, rfm_features, rfm_scaler):
    """
    Trains a dedicated model using only RFM features.
    This model is specifically for the dashboard's quick scoring feature.
    
    Uses a softer labeling approach to get smoother probability distributions.
    """
    from sklearn.linear_model import LogisticRegression
    
    try:
        print("\n=== Training Dedicated RFM Model ===")
        
        X_rfm = labeled_df[rfm_features].copy()
        y_original = labeled_df['RiskLabel'].copy()
        
        # Scale the RFM features
        X_rfm_scaled = pd.DataFrame(rfm_scaler.transform(X_rfm), columns=rfm_features)
        
        # Create a softer target based on RFM scores (not just cluster labels)
        # This helps the logistic regression produce smoother probabilities
        # Risk Score = High Recency (bad) - High Frequency (good) - High Monetary (good)
        risk_score = X_rfm_scaled['Recency'] - 0.3 * X_rfm_scaled['Frequency'] - 0.3 * X_rfm_scaled['Monetary_Total']
        
        # Convert to probability-like target using sigmoid
        # Add noise to prevent perfect separation
        np.random.seed(42)
        noise = np.random.normal(0, 0.3, len(risk_score))
        risk_score_noisy = risk_score + noise
        
        # Use quantile-based labeling for smoother boundaries
        # Bottom 40% = low risk (0), Top 40% = high risk (1), Middle 20% = mixed
        quantile_30 = risk_score_noisy.quantile(0.30)
        quantile_70 = risk_score_noisy.quantile(0.70)
        
        # For middle zone, randomly assign based on proximity
        y_soft = pd.Series(0, index=X_rfm_scaled.index)
        y_soft[risk_score_noisy > quantile_70] = 1
        
        # Middle zone gets probabilistic assignment
        middle_mask = (risk_score_noisy > quantile_30) & (risk_score_noisy <= quantile_70)
        middle_probs = (risk_score_noisy[middle_mask] - quantile_30) / (quantile_70 - quantile_30)
        np.random.seed(42)
        y_soft[middle_mask] = (np.random.random(middle_mask.sum()) < middle_probs).astype(int)
        
        print(f"Soft label distribution: {y_soft.value_counts().to_dict()}")
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_rfm_scaled, y_soft, test_size=0.2, stratify=y_soft, random_state=42
        )
        
        print(f"Training RFM model with {len(rfm_features)} features...")
        
        # Use LogisticRegression with lower regularization for smoother probabilities
        rfm_model = LogisticRegression(C=0.1, solver='lbfgs', random_state=42, max_iter=1000)
        rfm_model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = rfm_model.predict(X_test)
        y_pred_proba = rfm_model.predict_proba(X_test)[:, 1]
        
        metrics = eval_metrics(y_test, y_pred, y_pred_proba)
        print(f"RFM Model Performance: AUC={metrics['roc_auc']:.4f}, Accuracy={metrics['accuracy']:.4f}")
        
        # Show probability distribution
        print(f"Probability distribution on test set:")
        print(f"  Min: {y_pred_proba.min():.4f}, Max: {y_pred_proba.max():.4f}")
        print(f"  Mean: {y_pred_proba.mean():.4f}, Std: {y_pred_proba.std():.4f}")
        
        # Save the RFM-specific model
        pickle.dump(rfm_model, open("models/rfm_model.pkl", "wb"))
        print("✅ RFM model saved to models/rfm_model.pkl")
        
        # Log coefficients
        print("Feature Coefficients:")
        for feat, coef in zip(rfm_features, rfm_model.coef_[0]):
            print(f"  {feat}: {coef:.4f}")
        print(f"  Intercept: {rfm_model.intercept_[0]:.4f}")
        
        return rfm_model, metrics['roc_auc']
    except Exception as e:
        print(f"Warning: Could not train RFM model: {e}")
        import traceback
        traceback.print_exc()
        return None, 0.0

def main():
    # 1. Load Data
    data_path = "./data/processed/model_ready_data.csv"
    df = load_data(data_path)
    if df is None: return

    # 2. Save preprocessing pipeline (required for inference to work correctly)
    save_preprocessing_pipeline()
    
    # 2b. Save RFM scaler from RAW data and train dedicated RFM model
    rfm_scaler, labeled_df, rfm_features = save_rfm_scaler_from_raw()
    if rfm_scaler is not None and labeled_df is not None:
        train_rfm_model(labeled_df, rfm_features, rfm_scaler)
    
    # 3. Split Features and Target
    # Exclude ID columns and Target
    drop_cols = ['AccountId', 'RiskLabel', 'BatchId', 'SubscriptionId', 'CustomerId']
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df['RiskLabel']
    
    print(f"Training with {X.shape[1]} features.")
    
    # 4. Create and FIT ML pipeline (WoE + StandardScaler) on the actual data
    print("Creating and fitting ML pipeline (WoE + StandardScaler)...")
    ml_pipeline = create_ml_pipeline()
    ml_pipeline.fit(X, y)  # FIT the pipeline on training data
    save_ml_pipeline(ml_pipeline)
    
    # 5. Train-Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # 6. Transform data using the fitted ML pipeline
    print("Transforming training and test data with fitted ML pipeline...")
    X_train_transformed = ml_pipeline.transform(X_train)
    X_test_transformed = ml_pipeline.transform(X_test)
    
    # 7. Setup MLflow
    mlflow.set_experiment("Bati_Bank_Credit_Scoring")
    
    # --- Model 1: Logistic Regression (Baseline) ---
    lr_params = {
        'C': [0.1, 1.0, 10.0],
        'solver': ['liblinear']
    }
    auc_lr = train_model("LogisticRegression", LogisticRegression(max_iter=1000), 
                         lr_params, X_train_transformed, X_test_transformed, y_train, y_test)

    # --- Model 2: Random Forest (Challenger) ---
    rf_params = {
        'n_estimators': [50, 100],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5]
    }
    auc_rf = train_model("RandomForest", RandomForestClassifier(random_state=42), 
                         rf_params, X_train_transformed, X_test_transformed, y_train, y_test)

    # --- Compare and Print Best ---
    print("\n=== Experiment Summary ===")
    print(f"Logistic Regression AUC: {auc_lr:.4f}")
    print(f"Random Forest AUC:       {auc_rf:.4f}")
    
    if auc_rf > auc_lr:
        print(">> Recommendation: Deploy Random Forest")
    else:
        print(">> Recommendation: Deploy Logistic Regression (Simpler & Better)")

if __name__ == "__main__":
    main()