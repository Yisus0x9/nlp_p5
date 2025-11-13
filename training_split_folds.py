# -*- coding: utf-8 -*-
"""
Análisis de Sentimientos - Entrenamiento con K-Fold Cross Validation
Sentiment Analysis - Training with 5-Fold CV (GPU Optimized)
"""

import pandas as pd
from sklearn.model_selection import KFold, cross_validate
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, make_scorer)
import pickle
import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Librerías para GPU/CUDA
import torch
import cupy as cp
from cuml.linear_model import LogisticRegression as cuLogisticRegression
from cuml.svm import SVC as cuSVC
from cuml.naive_bayes import MultinomialNB as cuMultinomialNB
from cuml.ensemble import RandomForestClassifier as cuRandomForestClassifier

# Para CPU
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from scipy.sparse import issparse

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

MAX_FEATURES_GPU = 10000
USE_GPU = True
N_FOLDS = 5  # 5-Fold Cross Validation

print("=" * 80)
print("CONFIGURACIÓN")
print("=" * 80)
print(f"  MAX_FEATURES_GPU: {MAX_FEATURES_GPU}")
print(f"  USE_GPU: {USE_GPU}")
print(f"  N_FOLDS: {N_FOLDS}")

# ============================================================================
# VERIFICAR GPU/CUDA
# ============================================================================

cuda_available = torch.cuda.is_available() and USE_GPU
if cuda_available:
    print(f"\n✓ CUDA disponible")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"  Memoria GPU total: {gpu_memory:.2f} GB")
    torch.cuda.empty_cache()
    free_memory = torch.cuda.mem_get_info()[0] / 1e9
    print(f"  Memoria GPU libre: {free_memory:.2f} GB")
    device = 'cuda'
else:
    print("\n⚠ Usando CPU (CUDA deshabilitado o no disponible)")
    device = 'cpu'

print("=" * 80 + "\n")

# ============================================================================
# RUTAS
# ============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VECTORIZED_DIR = os.path.join(BASE_DIR, 'vectorized_datasets')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

os.makedirs(RESULTS_DIR, exist_ok=True)

if not os.path.exists(VECTORIZED_DIR):
    print(f"❌ ERROR: No se encontró {VECTORIZED_DIR}")
    exit(1)

# ============================================================================
# DESCUBRIR ARCHIVOS
# ============================================================================

print("PASO 1: Descubriendo archivos vectorizados...")

all_files = [f for f in os.listdir(VECTORIZED_DIR) if f.endswith('.pkl')]
train_files = [f for f in all_files if '_X_train.pkl' in f]
test_files = [f for f in all_files if '_X_test.pkl' in f]

combinations = []
for train_file in train_files:
    base_name = train_file.replace('_X_train.pkl', '')
    test_file = f"{base_name}_X_test.pkl"
    if test_file in test_files:
        parts = base_name.rsplit('_', 1)
        if len(parts) == 2:
            preprocessing, vectorization = parts
            combinations.append({
                'preprocessing': preprocessing,
                'vectorization': vectorization,
                'base_name': base_name
            })

print(f"✓ {len(combinations)} combinaciones encontradas\n")

# ============================================================================
# CARGAR ETIQUETAS
# ============================================================================

print("Cargando etiquetas...", end=' ')
with open(os.path.join(VECTORIZED_DIR, 'y_train.pkl'), 'rb') as f:
    y_train = pickle.load(f)
with open(os.path.join(VECTORIZED_DIR, 'y_test.pkl'), 'rb') as f:
    y_test = pickle.load(f)

# Convertir a array si es Series
if hasattr(y_train, 'values'):
    y_train = y_train.values
if hasattr(y_test, 'values'):
    y_test = y_test.values

print(f"✓ (Train: {len(y_train)}, Test: {len(y_test)})\n")

# ============================================================================
# FUNCIÓN PARA DETERMINAR SI USAR GPU
# ============================================================================

def can_use_gpu(X_train, X_test):
    """Determina si los datos caben en GPU"""
    if not cuda_available:
        return False
    
    n_features = X_train.shape[1]
    if n_features > MAX_FEATURES_GPU:
        return False
    
    if issparse(X_train):
        estimated_memory = (X_train.shape[0] * X_train.shape[1] * 8 + 
                          X_test.shape[0] * X_test.shape[1] * 8) / 1e9
    else:
        estimated_memory = (X_train.nbytes + X_test.nbytes) / 1e9
    
    free_memory = torch.cuda.mem_get_info()[0] / 1e9
    return estimated_memory < (free_memory - 2.0)

# ============================================================================
# CONFIGURAR MODELOS
# ============================================================================

def get_models_cpu():
    """Modelos para CPU"""
    return {
        'Logistic Regression': LogisticRegression(
            max_iter=500, 
            random_state=0, 
            n_jobs=-1,
            solver='saga'
        ),
        'Linear SVC': LinearSVC(
            max_iter=500, 
            random_state=0,
            dual='auto'
        ),
        'Multinomial NB': MultinomialNB(),
        'Random Forest': RandomForestClassifier(
            n_estimators=50, 
            max_depth=12,
            random_state=0, 
            n_jobs=-1
        )
    }

# ============================================================================
# K-FOLD CROSS VALIDATION MANUAL (CPU)
# ============================================================================

def cross_validate_cpu(model, X_train, y_train, n_folds=5):
    """K-Fold Cross Validation en CPU"""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=0)
    
    f1_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), 1):
        X_tr = X_train[train_idx]
        X_val = X_train[val_idx]
        y_tr = y_train[train_idx]
        y_val = y_train[val_idx]
        
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)
        
        f1 = f1_score(y_val, y_pred, average='macro', zero_division=0)
        f1_scores.append(f1)
    
    return np.mean(f1_scores)

# ============================================================================
# K-FOLD CROSS VALIDATION MANUAL (GPU)
# ============================================================================

def cross_validate_gpu(model_class, X_train_gpu, y_train_gpu, n_folds=5):
    """K-Fold Cross Validation en GPU"""
    n_samples = X_train_gpu.shape[0]
    indices = cp.arange(n_samples)
    cp.random.seed(0)
    cp.random.shuffle(indices)
    
    fold_size = n_samples // n_folds
    f1_scores = []
    
    for fold in range(n_folds):
        # Crear índices de validación y entrenamiento
        val_start = fold * fold_size
        val_end = (fold + 1) * fold_size if fold < n_folds - 1 else n_samples
        
        val_idx = indices[val_start:val_end]
        train_idx = cp.concatenate([indices[:val_start], indices[val_end:]])
        
        # Split datos
        X_tr = X_train_gpu[train_idx]
        X_val = X_train_gpu[val_idx]
        y_tr = y_train_gpu[train_idx]
        y_val = y_train_gpu[val_idx]
        
        # Entrenar modelo nuevo
        model = model_class
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)
        
        # Calcular F1 en CPU
        y_val_cpu = cp.asnumpy(y_val)
        y_pred_cpu = cp.asnumpy(y_pred)
        f1 = f1_score(y_val_cpu, y_pred_cpu, average='macro', zero_division=0)
        f1_scores.append(f1)
    
    return np.mean(f1_scores)

# ============================================================================
# ENTRENAMIENTO CON K-FOLD CV
# ============================================================================

print("=" * 80)
print("PASO 2: Entrenamiento con 5-Fold Cross Validation")
print("=" * 80)

results = []
exp_count = 0

for combo in combinations:
    prep = combo['preprocessing']
    vec = combo['vectorization']
    base_name = combo['base_name']
    
    print(f"\n{'─' * 80}")
    print(f"Datos: {prep} + {vec}")
    print(f"{'─' * 80}")
    
    # Cargar datos
    try:
        print(f"  Cargando datos...", end=' ')
        with open(os.path.join(VECTORIZED_DIR, f'{base_name}_X_train.pkl'), 'rb') as f:
            X_train_vec = pickle.load(f)
        with open(os.path.join(VECTORIZED_DIR, f'{base_name}_X_test.pkl'), 'rb') as f:
            X_test_vec = pickle.load(f)
        
        n_features = X_train_vec.shape[1]
        is_sparse = issparse(X_train_vec)
        print(f"✓ Shape: {X_train_vec.shape}, Features: {n_features}")
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        continue
    
    # Decidir dispositivo
    use_gpu_for_this = can_use_gpu(X_train_vec, X_test_vec)
    
    if cuda_available and not use_gpu_for_this:
        print(f"  ⚠ Demasiadas features ({n_features}). Usando CPU.")
    
    device_used = 'GPU' if use_gpu_for_this else 'CPU'
    print(f"  Dispositivo: {device_used}")
    
    # === USAR CPU ===
    if not use_gpu_for_this:
        models_config = get_models_cpu()
        
        for model_name, model in models_config.items():
            exp_count += 1
            print(f"  [{exp_count}] {model_name}...", end=' ')
            
            try:
                # K-Fold CV
                start_cv = datetime.now()
                avg_f1_cv = cross_validate_cpu(model, X_train_vec, y_train, n_folds=N_FOLDS)
                cv_time = (datetime.now() - start_cv).total_seconds()
                
                print(f"CV F1: {avg_f1_cv:.4f} ({cv_time:.1f}s)", end=' | ')
                
                # Entrenar en train completo
                start_train = datetime.now()
                model.fit(X_train_vec, y_train)
                train_time = (datetime.now() - start_train).total_seconds()
                
                # Evaluar en test
                y_pred = model.predict(X_test_vec)
                
                metrics = {
                    'model': model_name,
                    'preprocessing': prep,
                    'vectorization': vec,
                    'n_features': n_features,
                    'device': device_used,
                    'cv_f1_macro': avg_f1_cv,
                    'cv_time': cv_time,
                    'train_time': train_time,
                    'test_accuracy': accuracy_score(y_test, y_pred),
                    'test_f1_macro': f1_score(y_test, y_pred, average='macro', zero_division=0),
                    'test_precision_macro': precision_score(y_test, y_pred, average='macro', zero_division=0),
                    'test_recall_macro': recall_score(y_test, y_pred, average='macro', zero_division=0)
                }
                results.append(metrics)
                
                print(f"Test F1: {metrics['test_f1_macro']:.4f}, Train: {train_time:.1f}s")
                
            except Exception as e:
                print(f"✗ Error: {str(e)}")
    
    # === USAR GPU ===
    else:
        try:
            print(f"  Transfiriendo a GPU...", end=' ')
            
            if 'X_train_gpu' in locals():
                del X_train_gpu, X_test_gpu, y_train_gpu, y_test_gpu
            torch.cuda.empty_cache()
            cp.get_default_memory_pool().free_all_blocks()
            
            if is_sparse:
                X_train_dense = X_train_vec.toarray()
                X_test_dense = X_test_vec.toarray()
            else:
                X_train_dense = X_train_vec
                X_test_dense = X_test_vec
            
            X_train_gpu = cp.asarray(X_train_dense)
            X_test_gpu = cp.asarray(X_test_dense)
            y_train_gpu = cp.asarray(y_train)
            y_test_gpu = cp.asarray(y_test)
            
            del X_train_dense, X_test_dense
            
            memory_used = (X_train_gpu.nbytes + X_test_gpu.nbytes) / 1e9
            print(f"✓ {memory_used:.2f} GB")
            
            # Modelos GPU
            gpu_models = {
                'Logistic Regression (GPU)': cuLogisticRegression(max_iter=500),
                'Multinomial NB (GPU)': cuMultinomialNB(),
                'Random Forest (GPU)': cuRandomForestClassifier(n_estimators=50, max_depth=12, n_streams=1)
            }
            
            for model_name, model in gpu_models.items():
                exp_count += 1
                print(f"  [{exp_count}] {model_name}...", end=' ')
                
                try:
                    # K-Fold CV
                    start_cv = datetime.now()
                    avg_f1_cv = cross_validate_gpu(model, X_train_gpu, y_train_gpu, n_folds=N_FOLDS)
                    cv_time = (datetime.now() - start_cv).total_seconds()
                    
                    print(f"CV F1: {avg_f1_cv:.4f} ({cv_time:.1f}s)", end=' | ')
                    
                    # Entrenar en train completo
                    start_train = datetime.now()
                    model.fit(X_train_gpu, y_train_gpu)
                    train_time = (datetime.now() - start_train).total_seconds()
                    
                    # Evaluar en test
                    y_pred_gpu = model.predict(X_test_gpu)
                    y_pred = cp.asnumpy(y_pred_gpu)
                    y_test_cpu = cp.asnumpy(y_test_gpu)
                    
                    metrics = {
                        'model': model_name,
                        'preprocessing': prep,
                        'vectorization': vec,
                        'n_features': n_features,
                        'device': device_used,
                        'cv_f1_macro': avg_f1_cv,
                        'cv_time': cv_time,
                        'train_time': train_time,
                        'test_accuracy': accuracy_score(y_test_cpu, y_pred),
                        'test_f1_macro': f1_score(y_test_cpu, y_pred, average='macro', zero_division=0),
                        'test_precision_macro': precision_score(y_test_cpu, y_pred, average='macro', zero_division=0),
                        'test_recall_macro': recall_score(y_test_cpu, y_pred, average='macro', zero_division=0)
                    }
                    results.append(metrics)
                    
                    print(f"Test F1: {metrics['test_f1_macro']:.4f}, Train: {train_time:.1f}s")
                    
                except Exception as e:
                    print(f"✗ Error: {str(e)}")
            
            # Limpiar GPU
            del X_train_gpu, X_test_gpu, y_train_gpu, y_test_gpu
            torch.cuda.empty_cache()
            cp.get_default_memory_pool().free_all_blocks()
            
        except Exception as e:
            print(f"✗ Error GPU: {str(e)}")

print(f"\n{'=' * 80}")
print("✓ ENTRENAMIENTO COMPLETADO")
print(f"{'=' * 80}\n")

# ============================================================================
# GUARDAR RESULTADOS
# ============================================================================

if len(results) == 0:
    print("❌ No se obtuvieron resultados")
    exit(1)

results_df = pd.DataFrame(results)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
csv_path = os.path.join(RESULTS_DIR, f'results_kfold_{timestamp}.csv')
results_df.to_csv(csv_path, index=False)
print(f"✓ Resultados: {csv_path}\n")

# ============================================================================
# VISUALIZACIONES
# ============================================================================

print("Generando visualizaciones...")

plt.style.use('seaborn-v0_8-darkgrid')
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Sentiment Analysis - 5-Fold Cross Validation', fontsize=16, fontweight='bold')

# 1. CV F1-Score Heatmap
pivot_cv = results_df.pivot_table(values='cv_f1_macro', index='model', columns='vectorization', aggfunc='mean')
sns.heatmap(pivot_cv, annot=True, fmt='.3f', cmap='YlGnBu', ax=axes[0, 0])
axes[0, 0].set_title('CV F1-Score Macro (Promedio)')

# 2. Test F1-Score por preprocesamiento
prep_f1 = results_df.groupby(['preprocessing', 'model'])['test_f1_macro'].mean().reset_index()
top_preps = prep_f1.groupby('preprocessing')['test_f1_macro'].mean().nlargest(10).index
prep_f1_filtered = prep_f1[prep_f1['preprocessing'].isin(top_preps)]
sns.barplot(data=prep_f1_filtered, x='preprocessing', y='test_f1_macro', hue='model', ax=axes[0, 1])
axes[0, 1].set_title('Test F1-Score Macro (Top 10)')
axes[0, 1].tick_params(axis='x', rotation=45)

# 3. Mejor modelo
best = results_df.loc[results_df['cv_f1_macro'].idxmax()]
metrics_data = pd.DataFrame({
    'Métrica': ['CV F1', 'Test F1', 'Test Acc', 'Test Precision', 'Test Recall'],
    'Score': [best['cv_f1_macro'], best['test_f1_macro'], best['test_accuracy'], 
              best['test_precision_macro'], best['test_recall_macro']]
})
axes[1, 0].bar(range(len(metrics_data)), metrics_data['Score'])
axes[1, 0].set_xticks(range(len(metrics_data)))
axes[1, 0].set_xticklabels(metrics_data['Métrica'], rotation=45)
axes[1, 0].set_title(f'Mejor Modelo (CV): {best["model"][:25]}')
axes[1, 0].set_ylim(0, 1)

# 4. Top 10
top_10 = results_df.nlargest(10, 'cv_f1_macro')
top_10['config'] = top_10.apply(lambda x: f"{x['model'][:15]}\n{x['preprocessing'][:15]}", axis=1)
axes[1, 1].barh(range(len(top_10)), top_10['cv_f1_macro'])
axes[1, 1].set_yticks(range(len(top_10)))
axes[1, 1].set_yticklabels(top_10['config'], fontsize=8)
axes[1, 1].set_title('Top 10 (CV F1-Score)')
axes[1, 1].invert_yaxis()

plt.tight_layout()
plot_path = os.path.join(RESULTS_DIR, f'kfold_{timestamp}.png')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"✓ Gráfico: {plot_path}\n")

# ============================================================================
# REPORTE FINAL
# ============================================================================

print("=" * 80)
print("REPORTE FINAL - 5-FOLD CROSS VALIDATION")
print("=" * 80)

print(f"\n📊 ESTADÍSTICAS:")
print(f"  Experimentos: {len(results_df)}")
print(f"  Folds: {N_FOLDS}")

device_counts = results_df['device'].value_counts()
print(f"\n💻 DISPOSITIVOS:")
for dev, count in device_counts.items():
    print(f"  {dev}: {count} experimentos")

print(f"\n🏆 MEJOR MODELO (según CV F1-Score Macro):")
print(f"  Modelo: {best['model']}")
print(f"  Preprocesamiento: {best['preprocessing']}")
print(f"  Vectorización: {best['vectorization']}")
print(f"  Features: {best['n_features']}")
print(f"  Dispositivo: {best['device']}")
print(f"  CV F1-Score (macro): {best['cv_f1_macro']:.4f}")
print(f"  Test F1-Score (macro): {best['test_f1_macro']:.4f}")
print(f"  Test Accuracy: {best['test_accuracy']:.4f}")
print(f"  CV Time: {best['cv_time']:.2f}s")
print(f"  Train Time: {best['train_time']:.2f}s")

print(f"\n📈 TOP 5 CONFIGURACIONES (CV F1-Score):")
for idx, row in enumerate(top_10.head(5).itertuples(), 1):
    print(f"  {idx}. {row.model}")
    print(f"     Prep: {row.preprocessing} | Vec: {row.vectorization}")
    print(f"     CV F1: {row.cv_f1_macro:.4f} | Test F1: {row.test_f1_macro:.4f}")

print(f"\n⏱️ TIEMPOS POR MODELO:")
for model in sorted(results_df['model'].unique()):
    model_data = results_df[results_df['model'] == model]
    print(f"  {model}:")
    print(f"    Mejor CV F1: {model_data['cv_f1_macro'].max():.4f}")
    print(f"    Mejor Test F1: {model_data['test_f1_macro'].max():.4f}")
    print(f"    Tiempo CV promedio: {model_data['cv_time'].mean():.2f}s")
    print(f"    Tiempo Train promedio: {model_data['train_time'].mean():.2f}s")

print("\n" + "=" * 80)
print("✓ ANÁLISIS COMPLETADO")
print("=" * 80)
