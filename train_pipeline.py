import pandas as pd
import cupy as cp
import numpy as np
import os
import logging
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, ParameterGrid
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
import xgboost as xgb_core
from xgboost import XGBClassifier
import joblib
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json
import time
import traceback

# Configuration with enhanced options
CONFIG = {
    'csv_path': "mood_color_dataset_extended_v2.csv",
    'output_dir': "models",
    'pca_model_path': "pca_model.pkl",
    'xgb_model_path': "xgb_model.pkl",
    'scaler_model_path': "scaler_model.pkl",
    'encoder_model_path': "encoder_model.pkl",
    'version': '1.0.0',  # Add version
    'use_pca': True,  # Make PCA optional
    'n_components': 0,  # 0 means auto-determine based on explained_variance_ratio
    'pca_variance_threshold': 0.95,  # Retain 95% variance
    'viz_components': 2,  # Always create 2D visualization
    'test_size': 0.2,
    'random_state': 42,
    'perform_grid_search': True,
    'xgb_params': {
        'colsample_bytree': 1.0,
        'learning_rate': 0.1,
        'max_depth': 5,
        'n_estimators': 100,
        'subsample': 1.0,
        'device': 'cuda',
        'tree_method': 'hist',
    }
}

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def create_output_directory():
    """Create output directory if it doesn't exist"""
    if not os.path.exists(CONFIG['output_dir']):
        os.makedirs(CONFIG['output_dir'])
        logger.info(f"Created output directory: {CONFIG['output_dir']}")

try:
    import cupy as cp
    HAS_CUPY = True
    logger.info("CuPy is available for GPU acceleration")
except ImportError:
    HAS_CUPY = False
    logger.warning("CuPy not found. Install with: pip install cupy-cuda12x")

def check_gpu_support():
    """Check if XGBoost supports GPU and if GPU is available"""
    try:
        import xgboost as xgb
        build_info = xgb.build_info()
        
        logger.info("XGBoost build information:")
        for name in sorted(build_info.keys()):
            logger.info(f'{name}: {build_info[name]}')
        
        if 'USE_CUDA' in build_info and build_info['USE_CUDA']:
            logger.info("XGBoost has CUDA support enabled")
            return True
        else:
            logger.warning("XGBoost was not built with CUDA support")
            return False
    except Exception as e:
        logger.error(f"Error checking GPU support: {str(e)}")
        return False


def load_and_preprocess_data():
    """Load and preprocess the dataset with enhanced features"""
    logger.info(f"Loading data from {CONFIG['csv_path']}")
    
    try:
        df = pd.read_csv(CONFIG['csv_path'])
        logger.info(f"Dataset loaded with shape: {df.shape}")
        
        # Extract basic color features
        basic_feature_cols = ["r_mean", "g_mean", "b_mean", "r_std", "g_std", "b_std", 
                        "r_min", "g_min", "b_min", "r_max", "g_max", "b_max"]
        
        # Add histogram features if they don't exist in the dataset
        hist_feature_cols = []
        if 'r_hist_0' not in df.columns:
            logger.info("Computing histogram features from raw image data...")
            
            # Create histogram features (8 bins per channel)
            for channel in ['r', 'g', 'b']:
                for i in range(8):
                    col_name = f"{channel}_hist_{i}"
                    hist_feature_cols.append(col_name)
                    # Initialize with zeros
                    df[col_name] = 0.0
            
            # Compute histograms for each image
            # This assumes you have RGB values in your dataset
            # If you have image paths instead, you'll need to modify this part
            for idx, row in df.iterrows():
                for c_idx, channel in enumerate(['r', 'g', 'b']):
                    # Extract channel values from the dataset
                    # This assumes you have columns like r_values, g_values, b_values
                    # containing arrays or lists of pixel values
                    if f"{channel}_values" in df.columns:
                        channel_values = row[f"{channel}_values"]
                        if isinstance(channel_values, str):
                            # Convert string representation of array to actual array
                            channel_values = np.array(eval(channel_values))
                        
                        # Compute histogram with 8 bins
                        hist, _ = np.histogram(channel_values, bins=8, range=(0, 256), density=True)
                        
                        # Store histogram values
                        for i in range(8):
                            df.at[idx, f"{channel}_hist_{i}"] = hist[i]
                    else:
                        # If we don't have raw pixel values, compute approximate histograms
                        # from the statistics we do have (mean, std, min, max)
                        mean = row[f"{channel}_mean"]
                        std = row[f"{channel}_std"]
                        min_val = row[f"{channel}_min"]
                        max_val = row[f"{channel}_max"]
                        
                        # Generate a synthetic normal distribution based on stats
                        synthetic_values = np.random.normal(mean, std, 1000)
                        synthetic_values = np.clip(synthetic_values, min_val, max_val)
                        
                        # Compute histogram with 8 bins
                        hist, _ = np.histogram(synthetic_values, bins=8, range=(0, 256), density=True)
                        
                        # Store histogram values
                        for i in range(8):
                            df.at[idx, f"{channel}_hist_{i}"] = hist[i]
        else:
            # Histogram features already exist in the dataset
            for channel in ['r', 'g', 'b']:
                for i in range(8):
                    hist_feature_cols.append(f"{channel}_hist_{i}")
            logger.info("Using existing histogram features from dataset")
        
        # Combine all features
        all_feature_cols = basic_feature_cols + hist_feature_cols
        logger.info(f"Total features: {len(all_feature_cols)} (12 basic + {len(hist_feature_cols)} histogram)")
        
        X = df[all_feature_cols].values
        y = df["label"].values
        
        # Encode labels
        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(y)
        label_names = encoder.classes_
        
        # Save the encoder
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        encoder_path = os.path.join(CONFIG['output_dir'], 
                                   f"{CONFIG['version']}_{timestamp}_{CONFIG['encoder_model_path']}")
        joblib.dump(encoder, encoder_path)
        logger.info(f"Saved label encoder to {encoder_path}")
        
        # Save class-to-index mapping as JSON for reference
        class_mapping = {label: int(idx) for idx, label in enumerate(label_names)}
        mapping_path = os.path.join(CONFIG['output_dir'], 
                                   f"{CONFIG['version']}_{timestamp}_class_mapping.json")
        with open(mapping_path, 'w') as f:
            json.dump(class_mapping, f, indent=4)
        logger.info(f"Saved class-to-index mapping to {mapping_path}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, 
            test_size=CONFIG['test_size'], 
            random_state=CONFIG['random_state'],
            stratify=y_encoded  # Ensure balanced classes in train/test
        )
        
        logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test, label_names, all_feature_cols
        
    except Exception as e:
        logger.error(f"Error loading or preprocessing data: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def scale_and_reduce_features(X_train, X_test, y_train, feature_names):
    """Scale features and apply PCA with improved configuration"""
    try:
        # Feature Scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Save the scaler
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        scaler_path = os.path.join(
            CONFIG['output_dir'], 
            f"{CONFIG['version']}_{timestamp}_{CONFIG['scaler_model_path']}"
        )
        joblib.dump(scaler, scaler_path)
        logger.info(f"Saved scaler to {scaler_path}")
        
        # Initialize variables
        X_train_reduced = X_train_scaled
        X_test_reduced = X_test_scaled
        pca = None
        
        # Apply PCA if enabled
        if CONFIG['use_pca']:
            # Determine number of components
            if CONFIG['n_components'] <= 0:
                # Auto-determine based on variance threshold
                temp_pca = PCA(n_components=min(X_train_scaled.shape[0], X_train_scaled.shape[1]))
                temp_pca.fit(X_train_scaled)
                
                # Find number of components needed to reach variance threshold
                cumulative_variance = np.cumsum(temp_pca.explained_variance_ratio_)
                n_components = np.argmax(cumulative_variance >= CONFIG['pca_variance_threshold']) + 1
                logger.info(f"Auto-determined {n_components} PCA components to retain {CONFIG['pca_variance_threshold']*100:.1f}% variance")
            else:
                n_components = CONFIG['n_components']
                
            # Apply PCA for dimensionality reduction
            pca = PCA(n_components=n_components)
            X_train_reduced = pca.fit_transform(X_train_scaled)
            X_test_reduced = pca.transform(X_test_scaled)
            
            # Save the PCA model
            pca_path = os.path.join(
                CONFIG['output_dir'], 
                f"{CONFIG['version']}_{timestamp}_{CONFIG['pca_model_path']}"
            )
            joblib.dump(pca, pca_path)
            logger.info(f"Saved PCA model to {pca_path}")
            
            # Log explained variance
            explained_variance = pca.explained_variance_ratio_
            logger.info(f"PCA components: {n_components}")
            logger.info(f"PCA explained variance: {explained_variance}")
            logger.info(f"Total explained variance: {sum(explained_variance):.4f}")
        else:
            logger.info("PCA disabled, using scaled features directly")
        
        # Always create 2D PCA visualization
        viz_pca = PCA(n_components=CONFIG['viz_components'])
        X_train_viz = viz_pca.fit_transform(X_train_scaled)
        
        # Create visualization
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X_train_viz[:, 0], X_train_viz[:, 1], c=y_train, cmap='viridis', alpha=0.7)
        plt.colorbar(scatter)
        plt.title('PCA: First Two Principal Components (Visualization Only)')
        plt.xlabel(f'PC1 ({viz_pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({viz_pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.tight_layout()
        
        # Save with version and timestamp
        pca_viz_path = os.path.join(
            CONFIG['output_dir'], 
            f"{CONFIG['version']}_{timestamp}_pca_visualization.png"
        )
        plt.savefig(pca_viz_path)
        logger.info(f"Saved PCA visualization to {pca_viz_path}")
        
        # If using PCA, create feature importance visualization
        if CONFIG['use_pca'] and pca is not None:
            # Visualize feature contributions to principal components
            plt.figure(figsize=(12, 8))
            components = pd.DataFrame(
                pca.components_, 
                columns=feature_names,
                index=[f'PC{i+1}' for i in range(pca.n_components_)]
            )
            
            # Plot heatmap of component loadings
            sns.heatmap(components.iloc[:min(5, components.shape[0])], annot=False, cmap='coolwarm')
            plt.title('PCA Component Loadings (Top 5 Components)')
            plt.tight_layout()
            
            # Save with version and timestamp
            pca_components_path = os.path.join(
                CONFIG['output_dir'], 
                f"{CONFIG['version']}_{timestamp}_pca_components.png"
            )
            plt.savefig(pca_components_path)
            logger.info(f"Saved PCA components visualization to {pca_components_path}")
        
        return X_train_reduced, X_test_reduced, pca, scaler, n_components
        
    except Exception as e:
        logger.error(f"Error in scaling or PCA: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def train_model(X_train, y_train, X_test, y_test, label_names, feature_names, n_components, use_gpu=False):
    """Train the XGBoost model with versioning and metadata"""
    try:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        pca = None  # Initialize pca as None

        # Move data to GPU if using GPU acceleration
        if use_gpu:
            logger.info("Moving data to GPU for training")
            X_train_device = cp.asarray(X_train)
            y_train_device = cp.asarray(y_train)
            X_test_device = cp.asarray(X_test)
            y_test_device = cp.asarray(y_test)
        else:
            X_train_device, y_train_device = X_train, y_train
            X_test_device, y_test_device = X_test, y_test
        
        if CONFIG['perform_grid_search']:
            logger.info("Performing grid search for XGBoost parameters")
            # Expanded parameter grid for more thorough search
            param_grid = {
                'max_depth': [3, 5, 7, 9],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'n_estimators': [100, 200, 300],
                'colsample_bytree': [0.8, 0.9, 1.0],
                'subsample': [0.8, 0.9, 1.0],
                'gamma': [0, 0.1, 0.2]
            }
            
            xgb = XGBClassifier(
                eval_metric='mlogloss',
                random_state=CONFIG['random_state'],
                device='cuda' if use_gpu else 'cpu',
                tree_method='hist'
            )
            
            if use_gpu:
                logger.info("Using GPU-optimized approach for GridSearchCV")
                # Convert data to DMatrix format

                gdtrain = xgb_core.DMatrix(X_train_device, label=y_train_device)
                dtest = xgb_core.DMatrix(X_test_device, label=y_test_device)

                kf = KFold(n_splits=5, shuffle=True, random_state=CONFIG['random_state'])

                best_score = 0
                best_params = None

                logger.info("Starting grid search with {len(ParameterGrid(param_grid))} combinations - this may take some time...")
                
                for params in ParameterGrid(param_grid):
                    xgb_params = {
                        'max_depth': params['max_depth'],
                        'eta': params['learning_rate'],
                        'gamma': params['gamma'],
                        'colsample_bytree': params['colsample_bytree'],
                        'subsample': params['subsample'],
                        'objective': 'multi:softprob',
                        'num_class': len(label_names),
                        'eval_metric': 'mlogloss',
                        'device': 'cuda',
                        'tree_method': 'hist'
                    }

                    cv_scores = []

                    # Perform cross-validation
                    for train_idx, val_idx in kf.split(cp.asnumpy(X_train_device)):
                        # Extract train/val data for this fold
                        X_fold_train = X_train_device[train_idx]
                        y_fold_train = y_train_device[train_idx]
                        X_fold_val = X_train_device[val_idx]
                        y_fold_val = y_train_device[val_idx]
                        
                        # Create DMatrix objects
                        d_fold_train = xgb_core.DMatrix(X_fold_train, label=y_fold_train)
                        d_fold_val = xgb_core.DMatrix(X_fold_val, label=y_fold_val)
                        
                        # Train model
                        evallist = [(d_fold_train, 'train'), (d_fold_val, 'eval')]
                        fold_model = xgb_core.train(
                            xgb_params,
                            d_fold_train,
                            num_boost_round=params['n_estimators'],
                            evals=evallist,
                            verbose_eval=False
                        )

                        # Predict and calculate accuracy
                        y_pred = fold_model.predict(d_fold_val)
                        y_pred_labels = cp.argmax(y_pred, axis=1)
                        fold_acc = (y_pred_labels == y_fold_val).mean().item()
                        cv_scores.append(fold_acc)

                    # Calculate mean CV score
                    mean_cv_score = sum(cv_scores) / len(cv_scores)

                    # Update best parameters if better
                    if mean_cv_score > best_score:
                        best_score = mean_cv_score
                        best_params = params
                        logger.info(f"New best: {best_params} with score {best_score:.4f}")
                
                logger.info(f"Best parameters from grid search: {best_params}")

                # Train final model with best parameters
                final_params = {
                    'max_depth': best_params['max_depth'],
                    'eta': best_params['learning_rate'],
                    'gamma': best_params['gamma'],
                    'colsample_bytree': best_params['colsample_bytree'],
                    'subsample': best_params['subsample'],
                    'objective': 'multi:softprob',
                    'num_class': len(label_names),
                    'eval_metric': 'mlogloss',
                    'device': 'cuda',
                    'tree_method': 'hist'
                }

                # Train the final model
                evallist = [(dtrain, 'train'), (dtest, 'eval')]
                model_core = xgb_core.train(
                    final_params,
                    dtrain,
                    num_boost_round=best_params['n_estimators'],
                    evals=evallist,
                    verbose_eval=10
                )

                # Create a sklearn-compatible wrapper
                xgb = XGBClassifier(
                    max_depth=best_params['max_depth'],
                    learning_rate=best_params['learning_rate'],
                    n_estimators=best_params['n_estimators'],
                    colsample_bytree=best_params['colsample_bytree'],
                    subsample=best_params['subsample'],
                    gamma=best_params['gamma'],
                    device='cuda',
                    tree_method='hist'
                )

                # Set the booster
                xgb._Booster = model_core
            else:
                # Standard GridSearchCV for CPU mode
                grid_search = GridSearchCV(
                    xgb, param_grid, 
                    scoring='accuracy', 
                    cv=5, 
                    verbose=2,
                    n_jobs=-1
                )
                grid_search.fit(X_train_device, y_train_device)
                
                best_params = grid_search.best_params_
                logger.info(f"Best parameters from grid search: {best_params}")
            
            # Get best estimator
                xgb = grid_search.best_estimator_
            
            # Save best parameters
            best_params_path = os.path.join(
                CONFIG['output_dir'], 
                f"{CONFIG['version']}_{timestamp}_best_params.json"
            )
            with open(best_params_path, 'w') as f:
                json.dump(best_params, f, indent=4)
            
        else:
            logger.info("Training XGBoost with predefined parameters")
            # Initialize without use_label_encoder
            xgb = XGBClassifier(
                eval_metric='mlogloss',
                random_state=CONFIG['random_state'],
                device="cuda" if use_gpu else "cpu",
                tree_method="hist",  # Modern XGBoost uses 'hist' with device='cuda'
                **{k: v for k, v in CONFIG['xgb_params'].items() if k != 'use_label_encoder'}
            )

            # Direct training with GPU data
            if use_gpu:
                # Create DMatrix objects for better GPU performance
                import xgboost as xgb_core
                dtrain = xgb_core.DMatrix(X_train_device, label=y_train_device)
                dtest = xgb_core.DMatrix(X_test_device, label=y_test_device)
                
                # Train using the lower-level API for better GPU performance
                params = {
                    'max_depth': CONFIG['xgb_params'].get('max_depth', 5),
                    'learning_rate': CONFIG['xgb_params'].get('learning_rate', 0.1),
                    'objective': 'multi:softprob',
                    'num_class': len(label_names),
                    'eval_metric': 'mlogloss',
                    'device': 'cuda',
                    'tree_method': 'hist'
                }

                # Train the model
                evallist = [(dtrain, 'train'), (dtest, 'eval')]
                model_core = xgb_core.train(
                    params, 
                    dtrain, 
                    num_boost_round=CONFIG['xgb_params'].get('n_estimators', 100),
                    evals=evallist,
                    early_stopping_rounds=20,
                    verbose_eval=10
                )
                
                # Convert the core model to sklearn interface
                xgb.get_booster = lambda: model_core
            else:
                # Regular fit for CPU
                xgb.fit(X_train_device, y_train_device)
            
        
        # Save the model with version and timestamp
        xgb_path = os.path.join(
            CONFIG['output_dir'], 
            f"{CONFIG['version']}_{timestamp}_{CONFIG['xgb_model_path']}"
        )
        joblib.dump(xgb, xgb_path)
        logger.info(f"Saved XGBoost model to {xgb_path}")
        
        # Evaluate the model
        y_pred = xgb.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Get detailed classification report
        report = classification_report(y_test, y_pred, target_names=label_names, output_dict=True)
        report_text = classification_report(y_test, y_pred, target_names=label_names)
        
        logger.info(f"Model accuracy: {accuracy:.4f}")
        logger.info(f"Classification report:\n{report_text}")
        
        # Log macro/micro avg F1-scores
        macro_f1 = report['macro avg']['f1-score']
        micro_f1 = report['accuracy']
        weighted_f1 = report['weighted avg']['f1-score']
        
        logger.info(f"Macro-average F1 score: {macro_f1:.4f}")
        logger.info(f"Micro-average F1 score: {micro_f1:.4f}")
        logger.info(f"Weighted-average F1 score: {weighted_f1:.4f}")
        
        # Save classification report as JSON
        report_path = os.path.join(
            CONFIG['output_dir'], 
            f"{CONFIG['version']}_{timestamp}_classification_report.json"
        )
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=4)
        logger.info(f"Saved classification report to {report_path}")
        
        # Save classification report as text
        report_txt_path = os.path.join(
            CONFIG['output_dir'], 
            f"{CONFIG['version']}_{timestamp}_classification_report.txt"
        )
        with open(report_txt_path, 'w') as f:
            f.write(report_text)
        
        # Create confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_names, yticklabels=label_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        
        # Save confusion matrix with version and timestamp
        cm_path = os.path.join(
            CONFIG['output_dir'], 
            f"{CONFIG['version']}_{timestamp}_confusion_matrix.png"
        )
        plt.savefig(cm_path)
        logger.info(f"Saved confusion matrix to {cm_path}")
        
        # Feature importance analysis
        if hasattr(xgb, 'feature_importances_'):
            plt.figure(figsize=(12, 8))
            
            # Get feature importances
            importances = xgb.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            # Plot feature importances
            plt.bar(range(len(importances)), importances[indices])
            plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
            plt.title('Feature Importance')
            plt.tight_layout()
            
            # Save with version and timestamp
            importance_path = os.path.join(
                CONFIG['output_dir'], 
                f"{CONFIG['version']}_{timestamp}_feature_importance.png"
            )
            plt.savefig(importance_path)
            logger.info(f"Saved feature importance plot to {importance_path}")
        
        # Create metadata.json with all relevant information
        metadata = {
            "version": CONFIG['version'],
            "timestamp": timestamp,
            "model_type": "XGBoost",
            "performance": {
                "accuracy": float(accuracy),
                "macro_f1": float(macro_f1),
                "micro_f1": float(micro_f1),
                "weighted_f1": float(weighted_f1)
            },
            "training_config": {
                "use_pca": CONFIG['use_pca'],
                "pca_components": n_components if CONFIG['use_pca'] and pca is not None else 0,
                "random_state": CONFIG['random_state'],
                "test_size": CONFIG['test_size'],
                "feature_count": len(feature_names)
            },
            "model_params": xgb.get_params(),
            "class_labels": label_names.tolist(),
            "files": {
                "model": os.path.basename(xgb_path),
                "pca_model": f"{CONFIG['version']}_{timestamp}_{CONFIG['pca_model_path']}" if CONFIG['use_pca'] and pca is not None else None,
                "scaler_model": f"{CONFIG['version']}_{timestamp}_{CONFIG['scaler_model_path']}",
                "confusion_matrix": os.path.basename(cm_path),
                "classification_report": os.path.basename(report_path)
            }
        }
        
        # Save metadata
        metadata_path = os.path.join(
            CONFIG['output_dir'], 
            f"{CONFIG['version']}_{timestamp}_metadata.json"
        )
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        logger.info(f"Saved model metadata to {metadata_path}")
        
        return xgb, accuracy, report, macro_f1, micro_f1, weighted_f1
        
    except Exception as e:
        logger.error(f"Error in model training: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def train_and_evaluate():
    """Main function to train and evaluate the model with enhanced features"""
    try:
        start_time = time.time()
        logger.info(f"Starting training pipeline version {CONFIG['version']}")

        has_gpu_support = check_gpu_support()
        if has_gpu_support:
            logger.info("Using GPU acceleration for training")
        else:
            logger.warning("GPU support not available, falling back to CPU")
            # Update parameters to use CPU
            if 'device' in CONFIG['xgb_params']:
                CONFIG['xgb_params']['device'] = 'cpu'
            if 'tree_method' in CONFIG['xgb_params']:
                CONFIG['xgb_params']['tree_method'] = 'hist'
        
        # Create output directory
        create_output_directory()
        
        # Load and preprocess data with enhanced features
        X_train, X_test, y_train, y_test, label_names, feature_names = load_and_preprocess_data()
        
        # Scale and reduce features
        X_train_reduced, X_test_reduced, pca, scaler, n_components = scale_and_reduce_features(
            X_train, X_test, y_train, feature_names
        )
        
        # Train and evaluate model
        xgb, accuracy, report, macro_f1, micro_f1, weighted_f1 = train_model(
            X_train_reduced, y_train, X_test_reduced, y_test, label_names, feature_names, n_components
        )
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        time_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
        
        logger.info(f"Training pipeline v{CONFIG['version']} completed successfully in {time_str}")
        
        # Print summary
        print("\n" + "="*60)
        print(f"🎉 TRAINING PIPELINE v{CONFIG['version']} COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"📊 Model Accuracy: {accuracy:.4f}")
        print(f"📈 Macro-avg F1-score: {macro_f1:.4f}")
        print(f"📈 Micro-avg F1-score: {micro_f1:.4f}")
        print(f"📈 Weighted-avg F1-score: {weighted_f1:.4f}")
        print(f"🔢 Features: {len(feature_names)} ({12} basic + {len(feature_names)-12} histogram)")
        if CONFIG['use_pca']:
            if pca:
                print(f"🧮 PCA: {pca.n_components_} components ({sum(pca.explained_variance_ratio_):.2%} variance)")
            else:
                print("🧮 PCA: Enabled but not applied")
        else:
            print("🧮 PCA: Disabled")
        print(f"💾 Models saved to: {CONFIG['output_dir']}")
        print(f"⏱️ Total training time: {time_str}")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {str(e)}")
        print(f"\n❌ ERROR: Training pipeline failed: {str(e)}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    train_and_evaluate()
