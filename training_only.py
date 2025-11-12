# -*- coding: utf-8 -*-
"""
Análisis de Sentimientos - Solo Entrenamiento y Evaluación (GPU)
Sentiment Analysis - Training & Evaluation Only (Assumes vectorized data exists)
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

# Para fallback a CPU
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

# ============================================================================
# CONFIGURACIÓN GPU/CUDA
# ============================================================================

print("=" * 80)
print("VERIFICANDO DISPONIBILIDAD DE GPU/CUDA")
print("=" * 80)

cuda_available = torch.cuda.is_available()
if cuda_available:
    print(f"✓ CUDA disponible")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Memoria GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    device = 'cuda'
else:
    print("⚠ CUDA no disponible. Ejecutando en CPU con optimizaciones.")
    device = 'cpu'

print("=" * 80 + "\n")

# ============================================================================
# CONFIGURACIÓN DE RUTAS
# ============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VECTORIZED_DIR = os.path.join(BASE_DIR, 'vectorized_datasets')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# Crear directorio de resultados si no existe
os.makedirs(RESULTS_DIR, exist_ok=True)

print("CONFIGURACIÓN DEL PROYECTO")
print("=" * 80)
print(f"Directorio base: {BASE_DIR}")
print(f"Directorio de datasets vectorizados: {VECTORIZED_DIR}")
print(f"Directorio de resultados: {RESULTS_DIR}")
print(f"Dispositivo: {device.upper()}")
print("=" * 80 + "\n")

# Verificar que exista el directorio de vectorizados
if not os.path.exists(VECTORIZED_DIR):
    print(f"❌ ERROR: No se encontró el directorio {VECTORIZED_DIR}")
    print("Por favor, ejecuta primero el script de preprocesamiento y vectorización.")
    exit(1)

# ============================================================================
# PASO 1: DESCUBRIR ARCHIVOS VECTORIZADOS DISPONIBLES
# ============================================================================

print("PASO 1: Descubriendo archivos vectorizados...")
print("=" * 80)

# Buscar todos los archivos pkl en el directorio
all_files = [f for f in os.listdir(VECTORIZED_DIR) if f.endswith('.pkl')]

# Separar archivos de train y test
train_files = [f for f in all_files if '_X_train.pkl' in f]
test_files = [f for f in all_files if '_X_test.pkl' in f]

# Extraer combinaciones únicas (preprocesamiento + vectorización)
combinations = []
for train_file in train_files:
    # Extraer nombre base (sin _X_train.pkl)
    base_name = train_file.replace('_X_train.pkl', '')
    
    # Verificar que existe el archivo de test correspondiente
    test_file = f"{base_name}_X_test.pkl"
    if test_file in test_files:
        # Separar preprocesamiento y vectorización
        parts = base_name.rsplit('_', 1)
        if len(parts) == 2:
            preprocessing, vectorization = parts
            combinations.append({
                'preprocessing': preprocessing,
                'vectorization': vectorization,
                'base_name': base_name
            })

print(f"✓ Se encontraron {len(combinations)} combinaciones de datos vectorizados:")
print(f"  - Preprocesamiento único: {len(set(c['preprocessing'] for c in combinations))}")
print(f"  - Vectorización única: {len(set(c['vectorization'] for c in combinations))}")

# Verificar que existan las etiquetas
if 'y_train.pkl' not in all_files or 'y_test.pkl' not in all_files:
    print("❌ ERROR: No se encontraron los archivos y_train.pkl o y_test.pkl")
    exit(1)

print("✓ Archivos de etiquetas encontrados")
print("=" * 80 + "\n")

# ============================================================================
# PASO 2: CONFIGURAR MODELOS
# ============================================================================

print("PASO 2: Configurando modelos...")
print("=" * 80)

if cuda_available:
    print("✓ Usando modelos acelerados por GPU (cuML)\n")
    models_config = {
        'Logistic Regression (GPU)': cuLogisticRegression(max_iter=1000),
        'Linear SVC (GPU)': cuSVC(kernel='linear', max_iter=1000),
        'Multinomial NB (GPU)': cuMultinomialNB(),
        'Random Forest (GPU)': cuRandomForestClassifier(n_estimators=100, max_depth=16, n_streams=1)
    }
else:
    print("✓ Usando modelos CPU con paralelización\n")
    models_config = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1),
        'Linear SVC': LinearSVC(max_iter=1000, random_state=42),
        'Multinomial NB': MultinomialNB(),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    }

print(f"Modelos configurados: {len(models_config)}")
print("=" * 80 + "\n")

# ============================================================================
# PASO 3: ENTRENAMIENTO Y EVALUACIÓN
# ============================================================================

print("PASO 3: Entrenando y evaluando modelos...")
print("=" * 80)

results = []
total_experiments = len(combinations) * len(models_config)
exp_count = 0

# Cargar etiquetas una sola vez
print("Cargando etiquetas...", end=' ')
with open(os.path.join(VECTORIZED_DIR, 'y_train.pkl'), 'rb') as f:
    y_train = pickle.load(f)
with open(os.path.join(VECTORIZED_DIR, 'y_test.pkl'), 'rb') as f:
    y_test = pickle.load(f)
print(f"✓ (Train: {len(y_train)}, Test: {len(y_test)})\n")

for combo in combinations:
    prep = combo['preprocessing']
    vec = combo['vectorization']
    base_name = combo['base_name']
    
    print(f"\n{'─' * 80}")
    print(f"Datos: {prep} + {vec}")
    print(f"{'─' * 80}")
    
    # Cargar datos vectorizados
    try:
        print(f"  Cargando datos...", end=' ')
        with open(os.path.join(VECTORIZED_DIR, f'{base_name}_X_train.pkl'), 'rb') as f:
            X_train_vec = pickle.load(f)
        with open(os.path.join(VECTORIZED_DIR, f'{base_name}_X_test.pkl'), 'rb') as f:
            X_test_vec = pickle.load(f)
        print(f"✓ Shape: {X_train_vec.shape}")
    except Exception as e:
        print(f"✗ Error cargando: {str(e)}")
        continue
    
    # Convertir a GPU si está disponible
    if cuda_available:
        try:
            print(f"  Transfiriendo a GPU...", end=' ')
            # Convertir sparse matrix a dense array en GPU
            if hasattr(X_train_vec, 'toarray'):
                X_train_gpu = cp.array(X_train_vec.toarray())
                X_test_gpu = cp.array(X_test_vec.toarray())
            else:
                X_train_gpu = cp.array(X_train_vec)
                X_test_gpu = cp.array(X_test_vec)
            
            y_train_gpu = cp.array(y_train.values if hasattr(y_train, 'values') else y_train)
            y_test_gpu = cp.array(y_test.values if hasattr(y_test, 'values') else y_test)
            print(f"✓ GPU Memory used: {(X_train_gpu.nbytes + X_test_gpu.nbytes) / 1e9:.2f} GB")
        except Exception as e:
            print(f"✗ Error GPU: {str(e)}")
            print(f"  Fallback a CPU para este dataset")
            X_train_gpu = X_train_vec
            X_test_gpu = X_test_vec
            y_train_gpu = y_train
            y_test_gpu = y_test
            # Usar modelos CPU temporalmente
            temp_cuda = False
        else:
            temp_cuda = True
    else:
        X_train_gpu = X_train_vec
        X_test_gpu = X_test_vec
        y_train_gpu = y_train
        y_test_gpu = y_test
        temp_cuda = False
    
    # Entrenar cada modelo
    for model_name, model in models_config.items():
        exp_count += 1
        print(f"  [{exp_count}/{total_experiments}] {model_name}...", end=' ')
        
        try:
            # Entrenar
            start_time = datetime.now()
            model.fit(X_train_gpu, y_train_gpu)
            train_time = (datetime.now() - start_time).total_seconds()
            
            # Predecir
            y_pred = model.predict(X_test_gpu)
            
            # Convertir predicciones a CPU si es necesario
            if temp_cuda:
                y_pred_cpu = cp.asnumpy(y_pred)
                y_test_cpu = cp.asnumpy(y_test_gpu)
            else:
                y_pred_cpu = y_pred
                y_test_cpu = y_test_gpu
            
            # Calcular métricas
            metrics = {
                'model': model_name,
                'preprocessing': prep,
                'vectorization': vec,
                'accuracy': accuracy_score(y_test_cpu, y_pred_cpu),
                'precision_macro': precision_score(y_test_cpu, y_pred_cpu, average='macro', zero_division=0),
                'recall_macro': recall_score(y_test_cpu, y_pred_cpu, average='macro', zero_division=0),
                'f1_macro': f1_score(y_test_cpu, y_pred_cpu, average='macro', zero_division=0),
                'precision_weighted': precision_score(y_test_cpu, y_pred_cpu, average='weighted', zero_division=0),
                'recall_weighted': recall_score(y_test_cpu, y_pred_cpu, average='weighted', zero_division=0),
                'f1_weighted': f1_score(y_test_cpu, y_pred_cpu, average='weighted', zero_division=0),
                'training_time': train_time,
                'device': 'GPU' if temp_cuda else 'CPU'
            }
            results.append(metrics)
            
            print(f"Acc: {metrics['accuracy']:.4f}, F1: {metrics['f1_weighted']:.4f}, Time: {train_time:.2f}s")
            
        except Exception as e:
            print(f"✗ Error: {str(e)}")
            continue
    
    # Limpiar memoria GPU después de cada combinación
    if cuda_available:
        try:
            del X_train_gpu, X_test_gpu, y_train_gpu, y_test_gpu
            cp.get_default_memory_pool().free_all_blocks()
        except:
            pass

print(f"\n{'=' * 80}")
print("✓ ENTRENAMIENTO COMPLETADO")
print(f"{'=' * 80}\n")

# ============================================================================
# PASO 4: GUARDAR RESULTADOS
# ============================================================================

print("PASO 4: Guardando resultados...")

if len(results) == 0:
    print("❌ No se obtuvieron resultados. Revisa los errores anteriores.")
    exit(1)

results_df = pd.DataFrame(results)

# Guardar CSV con timestamp
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
csv_path = os.path.join(RESULTS_DIR, f'model_results_{device}_{timestamp}.csv')
results_df.to_csv(csv_path, index=False)
print(f"✓ Resultados guardados: {csv_path}")

# ============================================================================
# PASO 5: VISUALIZACIONES
# ============================================================================

print("\nPASO 5: Generando visualizaciones...")

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle(f'Comparación de Modelos ({device.upper()} Accelerated)', 
             fontsize=16, fontweight='bold')

# 1. Heatmap de accuracy
pivot_acc = results_df.pivot_table(values='accuracy', index='model', 
                                    columns='vectorization', aggfunc='mean')
sns.heatmap(pivot_acc, annot=True, fmt='.3f', cmap='YlGnBu', ax=axes[0, 0])
axes[0, 0].set_title('Accuracy Promedio por Modelo y Vectorización')
axes[0, 0].set_ylabel('Modelo')

# 2. F1-Score por preprocesamiento (Top 10)
prep_f1 = results_df.groupby(['preprocessing', 'model'])['f1_weighted'].mean().reset_index()
top_preps = prep_f1.groupby('preprocessing')['f1_weighted'].mean().nlargest(10).index
prep_f1_filtered = prep_f1[prep_f1['preprocessing'].isin(top_preps)]
sns.barplot(data=prep_f1_filtered, x='preprocessing', y='f1_weighted', hue='model', ax=axes[0, 1])
axes[0, 1].set_title('F1-Score por Preprocesamiento (Top 10)')
axes[0, 1].tick_params(axis='x', rotation=45)
axes[0, 1].set_xlabel('Preprocesamiento')
axes[0, 1].set_ylabel('F1-Score Weighted')
axes[0, 1].legend(loc='best', fontsize=8)

# 3. Métricas del mejor modelo
best = results_df.loc[results_df['f1_weighted'].idxmax()]
metrics_data = pd.DataFrame({
    'Métrica': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
    'Macro': [best['accuracy'], best['precision_macro'], best['recall_macro'], best['f1_macro']],
    'Weighted': [best['accuracy'], best['precision_weighted'], best['recall_weighted'], best['f1_weighted']]
})
x = np.arange(len(metrics_data))
axes[1, 0].bar(x - 0.2, metrics_data['Macro'], 0.4, label='Macro', alpha=0.8)
axes[1, 0].bar(x + 0.2, metrics_data['Weighted'], 0.4, label='Weighted', alpha=0.8)
axes[1, 0].set_xticks(x)
axes[1, 0].set_xticklabels(metrics_data['Métrica'])
axes[1, 0].set_title(f'Mejor Modelo: {best["model"]}\n{best["preprocessing"]} + {best["vectorization"]}')
axes[1, 0].legend()
axes[1, 0].set_ylabel('Score')
axes[1, 0].set_ylim(0, 1)

# 4. Top 10 configuraciones
top_10 = results_df.nlargest(10, 'f1_weighted')
top_10['config'] = top_10.apply(
    lambda x: f"{x['model'][:20]}\n{x['preprocessing'][:20]}\n{x['vectorization']}", 
    axis=1
)
colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_10)))
axes[1, 1].barh(range(len(top_10)), top_10['f1_weighted'], color=colors)
axes[1, 1].set_yticks(range(len(top_10)))
axes[1, 1].set_yticklabels(top_10['config'], fontsize=7)
axes[1, 1].set_title('Top 10 Configuraciones (F1-Score)')
axes[1, 1].set_xlabel('F1-Score Weighted')
axes[1, 1].invert_yaxis()
axes[1, 1].set_xlim(0, 1)

plt.tight_layout()
plot_path = os.path.join(RESULTS_DIR, f'model_comparison_{device}_{timestamp}.png')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"✓ Gráfico guardado: {plot_path}")

# ============================================================================
# PASO 6: REPORTE FINAL
# ============================================================================

print("\n" + "=" * 80)
print("REPORTE FINAL")
print("=" * 80)

print(f"\n📊 ESTADÍSTICAS GENERALES:")
print(f"  Total de experimentos: {len(results_df)}")
print(f"  Combinaciones evaluadas: {len(combinations)}")
print(f"  Modelos evaluados: {len(models_config)}")
print(f"  Dispositivo utilizado: {device.upper()}")

print(f"\n🏆 MEJOR CONFIGURACIÓN:")
print(f"  Modelo: {best['model']}")
print(f"  Preprocesamiento: {best['preprocessing']}")
print(f"  Vectorización: {best['vectorization']}")
print(f"  Accuracy: {best['accuracy']:.4f}")
print(f"  F1-Score (weighted): {best['f1_weighted']:.4f}")
print(f"  F1-Score (macro): {best['f1_macro']:.4f}")
print(f"  Precision (weighted): {best['precision_weighted']:.4f}")
print(f"  Recall (weighted): {best['recall_weighted']:.4f}")
print(f"  Tiempo de entrenamiento: {best['training_time']:.2f}s")

print(f"\n⏱️ TIEMPO TOTAL POR MODELO:")
for model in sorted(results_df['model'].unique()):
    model_data = results_df[results_df['model'] == model]
    total_time = model_data['training_time'].sum()
    avg_time = model_data['training_time'].mean()
    best_f1 = model_data['f1_weighted'].max()
    worst_f1 = model_data['f1_weighted'].min()
    print(f"  {model}:")
    print(f"    Total: {total_time:.2f}s | Avg: {avg_time:.2f}s")
    print(f"    Best F1: {best_f1:.4f} | Worst F1: {worst_f1:.4f}")

print(f"\n📈 TOP 5 CONFIGURACIONES:")
for idx, row in enumerate(top_10.head(5).itertuples(), 1):
    print(f"  {idx}. {row.model} | {row.preprocessing} | {row.vectorization}")
    print(f"     F1: {row.f1_weighted:.4f} | Acc: {row.accuracy:.4f} | Time: {row.training_time:.2f}s")

print(f"\n🔬 ANÁLISIS POR VECTORIZACIÓN:")
for vec in sorted(results_df['vectorization'].unique()):
    vec_data = results_df[results_df['vectorization'] == vec]
    print(f"  {vec}:")
    print(f"    Avg F1: {vec_data['f1_weighted'].mean():.4f}")
    print(f"    Best F1: {vec_data['f1_weighted'].max():.4f}")
    print(f"    Avg Time: {vec_data['training_time'].mean():.2f}s")

print(f"\n🔧 ANÁLISIS POR PREPROCESAMIENTO (Top 5):")
prep_stats = results_df.groupby('preprocessing').agg({
    'f1_weighted': ['mean', 'max'],
    'training_time': 'mean'
}).round(4)
prep_stats.columns = ['F1_Avg', 'F1_Max', 'Time_Avg']
prep_stats = prep_stats.sort_values('F1_Avg', ascending=False)
for prep, row in prep_stats.head(5).iterrows():
    print(f"  {prep}:")
    print(f"    Avg F1: {row['F1_Avg']:.4f} | Max F1: {row['F1_Max']:.4f} | Avg Time: {row['Time_Avg']:.2f}s")

print("\n" + "=" * 80)
print("✓ ANÁLISIS COMPLETO FINALIZADO")
print(f"✓ Archivos guardados en: {RESULTS_DIR}")
print("=" * 80)
