import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import numpy as np
import arff
import os
import time
import joblib

import model_manager

class MLPipeline:
    def __init__(self):
        self.dataset_path = self._find_dataset()
        self.df = None
        self.df_clean = None
        self.column_info = {}
        self.missing_counts = {}
        self.duplicate_count = 0
        self.file_size_kb = 0
        self.file_name = ""
        self.is_cleaned = False
        self.cleaning_report = None

    def _find_dataset(self):
        """Finds the path to the available dataset file."""
        dataset_dir = 'dataset'
        if os.path.exists(dataset_dir):
            for f in os.listdir(dataset_dir):
                if f.endswith('.arff') or f.endswith('.csv'):
                    return os.path.join(dataset_dir, f)
        return None

    def load_data(self):
        """Loads the raw dataset without any cleaning."""
        if not self.dataset_path:
            return False
        try:
            self.file_name = os.path.basename(self.dataset_path)
            self.file_size_kb = os.path.getsize(self.dataset_path) / 1024.0

            if self.dataset_path.endswith('.arff'):
                with open(self.dataset_path, 'r', encoding='utf-8') as f:
                    arff_data = arff.load(f)
                self.df = pd.DataFrame(arff_data['data'])
                self.df.columns = [attr[0] for attr in arff_data['attributes']]
                # ARFF can load data as bytes, decode them to strings if necessary
                str_cols = self.df.select_dtypes(include=['object']).columns
                self.df[str_cols] = self.df[str_cols].apply(lambda x: x.str.decode('utf-8') if isinstance(x.iloc[0], bytes) else x)
            else: # .csv
                self.df = pd.read_csv(self.dataset_path)

            self.missing_counts = self.df.isnull().sum().to_dict()
            self.duplicate_count = int(self.df.duplicated().sum())
            self.df.reset_index(drop=True, inplace=True)
            
            for col in self.df.columns:
                self.column_info[col] = str(self.df[col].dtype)
            return True
        except Exception: # Catch file not found and any parsing errors
            return False

    def get_dataset_info(self):
        """Returns column names and their data types."""
        if self.df is None:
            if not self.load_data():
                return {"error": "No dataset found or file is invalid. Please upload a .csv or .arff file."}, 404
        
        # Use cleaned dataset if available, otherwise use raw dataset
        dataset_to_use = self.df_clean if self.is_cleaned and self.df_clean is not None else self.df
            
        def safe_float(val):
            if pd.isna(val):
                return "-"
            return round(float(val), 4)

        feature_stats = {}
        for col in dataset_to_use.columns:
            is_num = pd.api.types.is_numeric_dtype(dataset_to_use[col])
            feature_stats[col] = {
                "type": str(dataset_to_use[col].dtype),
                "missing": int(dataset_to_use[col].isnull().sum()),
                "mean": safe_float(dataset_to_use[col].mean()) if is_num else "-",
                "min": safe_float(dataset_to_use[col].min()) if is_num else "-",
                "max": safe_float(dataset_to_use[col].max()) if is_num else "-"
            }

        # Generate dataset preview (first 5 rows)
        preview = dataset_to_use.head(5).where(pd.notnull(dataset_to_use.head(5)), None).to_dict(orient='records')
        total_missing = int(dataset_to_use.isnull().sum().sum())
        num_attrs = sum(pd.api.types.is_numeric_dtype(dataset_to_use[c]) for c in dataset_to_use.columns)
        cat_attrs = len(dataset_to_use.columns) - num_attrs

        unique_counts = {c: int(dataset_to_use[c].nunique(dropna=True)) for c in dataset_to_use.columns}

        # Feature Distributions for numeric attributes
        distributions = {}
        numeric_df = dataset_to_use.select_dtypes(include=np.number)
        for col in numeric_df.columns:
            clean_data = dataset_to_use[col].dropna()
            if not clean_data.empty:
                counts, bins = np.histogram(clean_data, bins=10)
                distributions[col] = {"counts": counts.tolist(), "bins": bins.tolist()}

        # Correlation Matrix
        corr_matrix = {}
        if not numeric_df.empty:
            corr_df = numeric_df.corr().fillna(0)
            corr_matrix = {"columns": corr_df.columns.tolist(), "values": corr_df.values.tolist()}
        
        # Update column_info to reflect cleaned dataset if cleaned
        if self.is_cleaned and self.df_clean is not None:
            self.column_info = {col: str(self.df_clean[col].dtype) for col in self.df_clean.columns}
        
        return {
            "fileName": self.file_name,
            "fileSizeKB": round(self.file_size_kb, 2),
            "columns": self.column_info,
            "rowCount": len(dataset_to_use),
            "features": feature_stats,
            "preview": preview,
            "totalMissing": total_missing,
            "duplicateRows": 0 if self.is_cleaned else self.duplicate_count,
            "numAttributes": num_attrs,
            "catAttributes": cat_attrs,
            "uniqueCounts": unique_counts,
            "distributions": distributions,
            "correlationMatrix": corr_matrix
        }

    def _validate_target_for_classification(self, y_raw, target_column):
        warnings = []

        # Drop missing targets for validation
        y_no_na = y_raw.dropna()
        if y_no_na.empty:
            raise ValueError(f"Target column '{target_column}' contains only missing values.")

        nunique = int(y_no_na.nunique(dropna=True))
        n = int(len(y_no_na))

        if nunique < 2:
            raise ValueError(f"Target column '{target_column}' must have at least 2 classes for classification.")

        # Warn if target looks like regression (many unique numeric values)
        if pd.api.types.is_numeric_dtype(y_no_na):
            # Heuristic: if a numeric target has many unique values relative to sample size
            if nunique >= min(20, max(10, int(0.5 * n))):
                warnings.append(
                    f"Target column '{target_column}' is numeric with {nunique} unique values. It may be better suited for regression than classification."
                )
        else:
            # If not numeric, it's likely categorical; still warn if extremely high cardinality
            if nunique >= min(50, max(25, int(0.5 * n))):
                warnings.append(
                    f"Target column '{target_column}' has high cardinality ({nunique} unique classes). Classification may be unstable."
                )

        return warnings

    def clean_dataset(self, target_column=None):
        """Performs cleaning on the raw dataset and stores the cleaned version.
        
        Args:
            target_column: Optional target column name to exclude from one-hot encoding
        """
        if self.df is None:
            if not self.load_data():
                raise ValueError("No dataset loaded.")

        # Step 1: Remove duplicate rows
        df_clean = self.df.drop_duplicates().copy()
        duplicates_removed = int(len(self.df) - len(df_clean))

        # Step 2: Fill missing numeric values with mean
        numeric_cols = df_clean.select_dtypes(include=np.number).columns
        missing_filled = 0
        for col in numeric_cols:
            missing_before = int(df_clean[col].isnull().sum())
            if missing_before > 0:
                mean_val = df_clean[col].mean()
                df_clean[col].fillna(mean_val, inplace=True)
                missing_filled += missing_before

        # Step 3: Encode categorical features (one-hot) - EXCLUDE target column if specified
        categorical_cols = df_clean.select_dtypes(include=['object', 'category']).columns
        # Exclude target column from encoding if provided
        if target_column and target_column in categorical_cols:
            categorical_cols = categorical_cols.drop(target_column)
        
        categorical_encoded = len(categorical_cols)
        if len(categorical_cols) > 0:
            df_clean = pd.get_dummies(df_clean, columns=categorical_cols, drop_first=False)

        # Step 4: Optional normalization (skipped for now; can be added later)
        normalized_features = 0

        self.df_clean = df_clean
        self.is_cleaned = True
        self.cleaning_report = {
            "duplicates_removed": duplicates_removed,
            "missing_values_filled": missing_filled,
            "categorical_columns_encoded": categorical_encoded,
            "features_normalized": normalized_features,
            "final_samples": int(len(df_clean)),
            "final_features": int(df_clean.shape[1])
        }
        return self.cleaning_report

    def save_cleaned_dataset(self):
        """Saves the cleaned dataset to disk for persistence."""
        if self.df_clean is None:
            return False
        try:
            dataset_dir = 'dataset'
            if not os.path.exists(dataset_dir):
                os.makedirs(dataset_dir)
            # Save cleaned dataset as CSV
            clean_path = os.path.join(dataset_dir, 'cleaned_dataset.csv')
            self.df_clean.to_csv(clean_path, index=False)
            # Save cleaning report as JSON
            import json
            report_path = os.path.join(dataset_dir, 'cleaning_report.json')
            with open(report_path, 'w') as f:
                json.dump(self.cleaning_report, f)
            return True
        except Exception as e:
            print(f"[Error] Failed to save cleaned dataset: {e}")
            return False

    def load_cleaned_dataset(self):
        """Loads the cleaned dataset from disk if it exists."""
        try:
            dataset_dir = 'dataset'
            clean_path = os.path.join(dataset_dir, 'cleaned_dataset.csv')
            report_path = os.path.join(dataset_dir, 'cleaning_report.json')
            
            if not os.path.exists(clean_path):
                return False
                
            # Load cleaned dataset
            self.df_clean = pd.read_csv(clean_path)
            self.is_cleaned = True
            
            # Load cleaning report if exists
            if os.path.exists(report_path):
                import json
                with open(report_path, 'r') as f:
                    self.cleaning_report = json.load(f)
            
            # Also load raw dataset for reference
            if self.df is None:
                self.load_data()
                
            return True
        except Exception as e:
            print(f"[Error] Failed to load cleaned dataset: {e}")
            return False

    def train(self, target_column, algorithm="GaussianNB", params=None, test_size=0.2):
        """Trains a Naive Bayes model using the dataset."""
        if self.df is None:
            if not self.load_data():
                raise ValueError("No dataset found. Please upload a dataset first.")

        # Prepare dataset for training
        df = self.df.copy()

        # Step 1: Handle missing values in target column
        df = df[df[target_column].notna()].copy()
        if len(df) < 5:
            raise ValueError("Dataset is too small.")

        # Prepare X and y
        y = df[target_column]
        X = df.drop(columns=[target_column])

        # Encode categorical features
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            X = pd.get_dummies(X, columns=categorical_cols, drop_first=False)

        # Impute missing numeric values
        numeric_cols = X.select_dtypes(include=np.number).columns
        if len(numeric_cols) > 0:
            imputer = SimpleImputer(strategy='mean')
            X[numeric_cols] = imputer.fit_transform(X[numeric_cols])

        # Encode target
        le = LabelEncoder()
        y_encoded = le.fit_transform(y.astype(str))
        class_names = [str(c) for c in le.classes_]

        # Naive Bayes Model
        model = GaussianNB()
        
        # Scaling
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

        # Split
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=test_size, random_state=42)
        
        # Train
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time

        # Eval
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        cm = confusion_matrix(y_test, y_pred).tolist()

        # NB Specifics
        nb_priors = dict(zip(class_names, model.class_prior_.tolist()))
        feature_means = []
        for i, feature in enumerate(X.columns):
            means = {cls: float(model.theta_[j][i]) for j, cls in enumerate(class_names)}
            feature_means.append({"feature": feature, "means": means})

        # ROC/AUC
        roc_data = None
        auc_score = None
        if len(class_names) == 2:
            try:
                y_prob = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                auc_score = float(roc_auc_score(y_test, y_prob))
                roc_data = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
            except: pass

        results = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "confusion_matrix": cm,
            "nb_priors": nb_priors,
            "nb_likelihoods": feature_means,
            "training_time": float(training_time),
            "auc_score": auc_score,
            "roc_data": roc_data,
            "algorithm": "GaussianNB"
        }

        metadata = {
            "target_column": target_column,
            "feature_columns": [str(col) for col in X.columns],
            "target_classes": class_names,
            "accuracy": float(accuracy),
            "algorithm": "GaussianNB"
        }
        model_manager.save_model(model, metadata, scaler=scaler)
        return results

    def get_cleaning_status(self):
        """Returns whether the dataset is cleaned and the cleaning report."""
        return {
            "is_cleaned": self.is_cleaned,
            "cleaning_report": self.cleaning_report
        }

    def predict(self, input_data, return_neighbors=False):
        """Makes a prediction using the trained model."""
        model, metadata, scaler = model_manager.load_model()
        if model is None:
            raise FileNotFoundError("Model not found. Please train a model first.")

        # Prepare input data as dictionary to ensure column alignment
        processed_input = {}
        for col in metadata['feature_columns']:
            # Get value from input_data, default to 0 if missing (or handle appropriately)
            val = input_data.get(col)
            if val is None:
                # Fallback to mean value if possible, or 0
                processed_input[col] = 0.0
            else:
                processed_input[col] = float(val)

        input_df = pd.DataFrame([processed_input], columns=metadata['feature_columns'])
        
        # Apply scaling if available
        if scaler:
            input_df = pd.DataFrame(scaler.transform(input_df), columns=input_df.columns)
        
        # Core prediction
        prediction_encoded = model.predict(input_df)
        target_classes = metadata.get('target_classes', [])
        prediction_idx = int(prediction_encoded[0])
        prediction_label = target_classes[prediction_idx] if prediction_idx < len(target_classes) else str(prediction_idx)

        # Get probability if available
        probability = 0.0
        confidence = "N/A"
        try:
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(input_df)[0]
                probability = float(np.max(probs))
                
                if probability > 0.8:
                    confidence = "High"
                elif probability > 0.6:
                    confidence = "Medium"
                else:
                    confidence = "Low"
        except Exception as e:
            print(f"[Warning] Could not calculate probability: {e}")

        result = {
            "prediction": prediction_idx,
            "prediction_label": str(prediction_label),
            "probability": probability,
            "confidence": confidence
        }
        
        return result
