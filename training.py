# -*- coding: utf-8 -*-
"""
Análisis de Sentimientos - Solo Entrenamiento y Evaluación (GPU Optimizado)
Sentiment Analysis - Training & Evaluation Only (Memory Optimized)
"""

import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix)
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

# Para fallback a CPU y manejo de sparse matrices
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from scipy.sparse import issparse

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

# IMPORTANTE: Ajusta estos parámetros según tu GPU
MAX_FEATURES_GPU = 44735  # Máximo de features para procesar en GPU
USE_GPU = True            # Cambiar a False para forzar CPU
BATCH_TRAINING = True     # Usar entrenamiento por lotes para modelos grandes

print("=" * 80)
print("CONFIGURACIÓN")
print("=" * 80)
print(f"  MAX_FEATURES_GPU: {MAX_FEATURES_GPU}")
print(f"  USE_GPU: {USE_GPU}")
print(f"  BATCH_TRAINING: {BATCH_TRAINING}")

# ============================================================================
# VERIFICAR GPU/CUDA
# ============================================================================

cuda_available = torch.cuda.is_available() and USE_GPU
if cuda_available:
    print(f"\n✓ CUDA disponible")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"  Memoria GPU total: {gpu_memory:.2f} GB")
    
    # Calcular memoria disponible
    torch.cuda.empty_cache()
    free_memory = torch.cuda.mem_get_info()[0] / 1e9
    print(f"  Memoria GPU libre: {free_memory:.2f} GB")
    device = 'cuda'
else:
    print("\n⚠ Usando CPU (CUDA deshabilitado o no disponible)")
    device = 'cpu'

print("=" * 80 + "\n")

# ============================================================================
# CONFIGURACIÓN DE RUTAS
# ============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VECTORIZED_DIR = os.path.join(BASE_DIR, 'vectorized_datasets')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

os.makedirs(RESULTS_DIR, exist_ok=True)

if not os.path.exists(VECTORIZED_DIR):
    print(f"❌ ERROR: No se encontró el directorio {VECTORIZED_DIR}")
    exit(1)

# ============================================================================
# DESCUBRIR ARCHIVOS VECTORIZADOS
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
print(f"✓ (Train: {len(y_train)}, Test: {len(y_test)})\n")

# ============================================================================
# FUNCIÓN PARA DETERMINAR SI USAR GPU
# ============================================================================

def can_use_gpu(X_train, X_test):
    """Determina si los datos caben en GPU"""
    if not cuda_available:
        return False
    
    # Obtener número de features
    n_features = X_train.shape[1]
    
    # Si hay demasiadas features, usar CPU
    if n_features > MAX_FEATURES_GPU:
        return False
    
    # Estimar memoria necesaria (aproximado)
    if issparse(X_train):
        # Para sparse, estimar memoria del array denso
        estimated_memory = (X_train.shape[0] * X_train.shape[1] * 8 + 
                          X_test.shape[0] * X_test.shape[1] * 8) / 1e9
    else:
        estimated_memory = (X_train.nbytes + X_test.nbytes) / 1e9
    
    # Dejar al menos 2GB libres
    free_memory = torch.cuda.mem_get_info()[0] / 1e9
    
    return estimated_memory < (free_memory - 2.0)

# ============================================================================
# CONFIGURAR MODELOS
# ============================================================================

def get_models(use_gpu_models):
    """Retorna configuración de modelos según disponibilidad"""
    if use_gpu_models:
        return {
            'Logistic Regression (GPU)': cuLogisticRegression(max_iter=500),
            'Multinomial NB (GPU)': cuMultinomialNB(),
            'Random Forest (GPU)': cuRandomForestClassifier(
                n_estimators=50, 
                max_depth=12,
                n_streams=1
            )
        }
    else:
        return {
            'Logistic Regression': LogisticRegression(
                max_iter=500, 
                random_state=42, 
                n_jobs=-1,
                solver='saga'
            ),
            'Linear SVC': LinearSVC(
                max_iter=500, 
                random_state=42,
                dual='auto'
            ),
            'Multinomial NB': MultinomialNB(),
            'Random Forest': RandomForestClassifier(
                n_estimators=50, 
                max_depth=12,
                random_state=42, 
                n_jobs=-1
            )
        }

# ============================================================================
# ENTRENAMIENTO
# ============================================================================

print("=" * 80)
print("PASO 2: Entrenando y evaluando modelos...")
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
        print(f"✓ Shape: {X_train_vec.shape}, Sparse: {is_sparse}, Features: {n_features}")
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        continue
    
    # Decidir si usar GPU
    use_gpu_for_this = can_use_gpu(X_train_vec, X_test_vec)
    
    if cuda_available and not use_gpu_for_this:
        print(f"  ⚠ Demasiadas features ({n_features}). Usando CPU para este dataset.")
    
    # Configurar modelos
    models_config = get_models(use_gpu_for_this)
    device_used = 'GPU' if use_gpu_for_this else 'CPU'
    print(f"  Dispositivo: {device_used}")
    
    # Preparar datos según el dispositivo
    if use_gpu_for_this:
        try:
            print(f"  Transfiriendo a GPU...", end=' ')
            
            # Limpiar memoria GPU primero
            if 'X_train_gpu' in locals():
                del X_train_gpu, X_test_gpu, y_train_gpu, y_test_gpu
            torch.cuda.empty_cache()
            cp.get_default_memory_pool().free_all_blocks()
            
            # Convertir a denso solo si es razonablemente pequeño
            if is_sparse:
                X_train_dense = X_train_vec.toarray()
                X_test_dense = X_test_vec.toarray()
            else:
                X_train_dense = X_train_vec
                X_test_dense = X_test_vec
            
            # Transferir a GPU
            X_train_gpu = cp.asarray(X_train_dense)
            X_test_gpu = cp.asarray(X_test_dense)
            y_train_gpu = cp.asarray(y_train.values if hasattr(y_train, 'values') else y_train)
            y_test_gpu = cp.asarray(y_test.values if hasattr(y_test, 'values') else y_test)
            
            # Liberar memoria CPU
            del X_train_dense, X_test_dense
            
            memory_used = (X_train_gpu.nbytes + X_test_gpu.nbytes) / 1e9
            print(f"✓ GPU Memory: {memory_used:.2f} GB")
            
            X_train_final = X_train_gpu
            X_test_final = X_test_gpu
            y_train_final = y_train_gpu
            y_test_final = y_test_gpu
            
        except Exception as e:
            print(f"✗ Error GPU: {str(e)}")
            print(f"  Fallback a CPU")
            use_gpu_for_this = False
            device_used = 'CPU'
            models_config = get_models(False)
            X_train_final = X_train_vec
            X_test_final = X_test_vec
            y_train_final = y_train
            y_test_final = y_test
    else:
        X_train_final = X_train_vec
        X_test_final = X_test_vec
        y_train_final = y_train
        y_test_final = y_test
    
    # Entrenar modelos
    total_models = len(models_config)
    for idx, (model_name, model) in enumerate(models_config.items(), 1):
        exp_count += 1
        print(f"  [{idx}/{total_models}] {model_name}...", end=' ')
        
        try:
            start_time = datetime.now()
            model.fit(X_train_final, y_train_final)
            train_time = (datetime.now() - start_time).total_seconds()
            
            y_pred = model.predict(X_test_final)
            
            # Convertir a CPU si es necesario
            if use_gpu_for_this:
                y_pred_cpu = cp.asnumpy(y_pred)
                y_test_cpu = cp.asnumpy(y_test_final)
            else:
                y_pred_cpu = y_pred
                y_test_cpu = y_test_final
            
            # Calcular métricas
            metrics = {
                'model': model_name,
                'preprocessing': prep,
                'vectorization': vec,
                'n_features': n_features,
                'accuracy': accuracy_score(y_test_cpu, y_pred_cpu),
                'precision_macro': precision_score(y_test_cpu, y_pred_cpu, average='macro', zero_division=0),
                'recall_macro': recall_score(y_test_cpu, y_pred_cpu, average='macro', zero_division=0),
                'f1_macro': f1_score(y_test_cpu, y_pred_cpu, average='macro', zero_division=0),
                'precision_weighted': precision_score(y_test_cpu, y_pred_cpu, average='weighted', zero_division=0),
                'recall_weighted': recall_score(y_test_cpu, y_pred_cpu, average='weighted', zero_division=0),
                'f1_weighted': f1_score(y_test_cpu, y_pred_cpu, average='weighted', zero_division=0),
                'training_time': train_time,
                'device': device_used
            }
            results.append(metrics)
            
            print(f"Acc: {metrics['accuracy']:.4f}, F1: {metrics['f1_weighted']:.4f}, Time: {train_time:.2f}s")
            
        except Exception as e:
            print(f"✗ Error: {str(e)}")
            continue
    
    # Limpiar memoria después de cada combinación
    if use_gpu_for_this:
        try:
            del X_train_gpu, X_test_gpu, y_train_gpu, y_test_gpu
            del X_train_final, X_test_final, y_train_final, y_test_final
            torch.cuda.empty_cache()
            cp.get_default_memory_pool().free_all_blocks()
        except:
            pass

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
csv_path = os.path.join(RESULTS_DIR, f'results_{timestamp}.csv')
results_df.to_csv(csv_path, index=False)
print(f"✓ Resultados guardados: {csv_path}\n")

# ============================================================================
# VISUALIZACIONES
# ============================================================================

print("Generando visualizaciones...")

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Comparación de Modelos - Sentiment Analysis', fontsize=16, fontweight='bold')

# 1. Heatmap
pivot_acc = results_df.pivot_table(values='accuracy', index='model', columns='vectorization', aggfunc='mean')
sns.heatmap(pivot_acc, annot=True, fmt='.3f', cmap='YlGnBu', ax=axes[0, 0])
axes[0, 0].set_title('Accuracy Promedio')

# 2. F1 por preprocesamiento
prep_f1 = results_df.groupby(['preprocessing', 'model'])['f1_weighted'].mean().reset_index()
top_preps = prep_f1.groupby('preprocessing')['f1_weighted'].mean().nlargest(10).index
prep_f1_filtered = prep_f1[prep_f1['preprocessing'].isin(top_preps)]
sns.barplot(data=prep_f1_filtered, x='preprocessing', y='f1_weighted', hue='model', ax=axes[0, 1])
axes[0, 1].set_title('F1-Score Top 10 Preprocesamiento')
axes[0, 1].tick_params(axis='x', rotation=45)

# 3. Mejor modelo
best = results_df.loc[results_df['f1_weighted'].idxmax()]
metrics_data = pd.DataFrame({
    'Métrica': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
    'Macro': [best['accuracy'], best['precision_macro'], best['recall_macro'], best['f1_macro']],
    'Weighted': [best['accuracy'], best['precision_weighted'], best['recall_weighted'], best['f1_weighted']]
})
x = np.arange(len(metrics_data))
axes[1, 0].bar(x - 0.2, metrics_data['Macro'], 0.4, label='Macro')
axes[1, 0].bar(x + 0.2, metrics_data['Weighted'], 0.4, label='Weighted')
axes[1, 0].set_xticks(x)
axes[1, 0].set_xticklabels(metrics_data['Métrica'])
axes[1, 0].set_title(f'Mejor: {best["model"][:25]}')
axes[1, 0].legend()
axes[1, 0].set_ylim(0, 1)

# 4. Top 10
top_10 = results_df.nlargest(10, 'f1_weighted')
top_10['config'] = top_10.apply(lambda x: f"{x['model'][:15]}\n{x['preprocessing'][:15]}", axis=1)
axes[1, 1].barh(range(len(top_10)), top_10['f1_weighted'])
axes[1, 1].set_yticks(range(len(top_10)))
axes[1, 1].set_yticklabels(top_10['config'], fontsize=8)
axes[1, 1].set_title('Top 10 Configuraciones')
axes[1, 1].invert_yaxis()

plt.tight_layout()
plot_path = os.path.join(RESULTS_DIR, f'comparison_{timestamp}.png')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"✓ Gráfico: {plot_path}\n")

# ============================================================================
# REPORTE FINAL
# ============================================================================

print("=" * 80)
print("REPORTE FINAL")
print("=" * 80)

print(f"\n📊 ESTADÍSTICAS:")
print(f"  Experimentos: {len(results_df)}")
print(f"  Combinaciones: {len(combinations)}")

# Contar por dispositivo
device_counts = results_df['device'].value_counts()
print(f"\n💻 DISPOSITIVOS USADOS:")
for dev, count in device_counts.items():
    print(f"  {dev}: {count} experimentos")

print(f"\n🏆 MEJOR CONFIGURACIÓN:")
print(f"  Modelo: {best['model']}")
print(f"  Preprocesamiento: {best['preprocessing']}")
print(f"  Vectorización: {best['vectorization']}")
print(f"  Features: {best['n_features']}")
print(f"  Dispositivo: {best['device']}")
print(f"  Accuracy: {best['accuracy']:.4f}")
print(f"  F1-Score: {best['f1_weighted']:.4f}")
print(f"  Tiempo: {best['training_time']:.2f}s")

print(f"\n⏱️ TIEMPOS POR MODELO:")
for model in sorted(results_df['model'].unique()):
    model_data = results_df[results_df['model'] == model]
    print(f"  {model}:")
    print(f"    Total: {model_data['training_time'].sum():.2f}s")
    print(f"    Promedio: {model_data['training_time'].mean():.2f}s")
    print(f"    Mejor F1: {model_data['f1_weighted'].max():.4f}")

print("\n" + "=" * 80)
print("✓ ANÁLISIS COMPLETADO")
print("=" * 80)
