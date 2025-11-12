# -*- coding: utf-8 -*-
"""
Análisis de Sentimientos - Pipeline Completo (GPU Accelerated con CUDA)
Sentiment Analysis Pipeline - GPU/CUDA Optimized
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix)

import nltk
import spacy
import pickle
import os
import re
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Librerías para GPU/CUDA
import torch
from multiprocessing import Pool, cpu_count
from functools import partial
import cupy as cp  # Versión GPU de numpy
from cuml.linear_model import LogisticRegression as cuLogisticRegression
from cuml.svm import SVC as cuSVC
from cuml.naive_bayes import MultinomialNB as cuMultinomialNB
from cuml.ensemble import RandomForestClassifier as cuRandomForestClassifier
from scipy.sparse import csr_matrix
import cupyx.scipy.sparse as cusp

# ============================================================================
# CONFIGURACIÓN GPU/CUDA
# ============================================================================

print("=" * 80)
print("VERIFICANDO DISPONIBILIDAD DE GPU/CUDA")
print("=" * 80)

# Verificar CUDA
cuda_available = torch.cuda.is_available()
if cuda_available:
    print(f"✓ CUDA disponible")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Memoria GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"  CuPy versión: {cp.__version__}")
    device = 'cuda'
else:
    print("⚠ CUDA no disponible. Ejecutando en CPU con optimizaciones.")
    device = 'cpu'

# Configurar número de threads para CPU
n_jobs = cpu_count()
print(f"  CPU cores disponibles: {n_jobs}")
print("=" * 80 + "\n")

# ============================================================================
# CONFIGURACIÓN DE RUTAS
# ============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
VECTORIZED_DIR = os.path.join(BASE_DIR, 'vectorized_datasets')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
DATASET_PATH = os.path.join(DATA_DIR, 'Rest_Mex_2022.xlsx')

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(VECTORIZED_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

print("CONFIGURACIÓN DEL PROYECTO")
print("=" * 80)
print(f"Directorio base: {BASE_DIR}")
print(f"Ruta del dataset: {DATASET_PATH}")
print(f"Dispositivo: {device.upper()}")
print("=" * 80 + "\n")

# ============================================================================
# PASO 1: CONFIGURACIÓN INICIAL
# ============================================================================

print("PASO 1: Configurando entorno...")

# Descargar recursos de NLTK
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("  Descargando stopwords de NLTK...")
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("  Descargando punkt de NLTK...")
    nltk.download('punkt', quiet=True)

# Cargar modelo de spaCy con GPU si está disponible
try:
    if cuda_available:
        spacy.require_gpu()
        print("  ✓ spaCy configurado para usar GPU")
    nlp = spacy.load('es_core_news_sm')
    print("  ✓ Modelo de spaCy cargado correctamente")
except:
    print("  ⚠ Modelo de spaCy no encontrado. Instalando...")
    os.system('python -m spacy download es_core_news_sm')
    nlp = spacy.load('es_core_news_sm')

print("✓ Configuración inicial completada\n")

# ============================================================================
# PASO 2: CARGA Y PREPARACIÓN DE DATOS
# ============================================================================

print("PASO 2: Cargando dataset...")

if not os.path.exists(DATASET_PATH):
    print(f"❌ ERROR: No se encontró el archivo {DATASET_PATH}")
    exit(1)

df = pd.read_excel(DATASET_PATH)
print(f"✓ Dataset cargado. Shape: {df.shape}")

df['texto_completo'] = df['Title'].astype(str).fillna('') + ' ' + df['Opinion'].astype(str).fillna('')
X = df['texto_completo']
y = df['Polarity']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0, shuffle=True
)

print(f"✓ Datos divididos: Train={X_train.shape[0]}, Test={X_test.shape[0]}\n")

# ============================================================================
# PASO 3: FUNCIONES DE PREPROCESAMIENTO OPTIMIZADAS
# ============================================================================

print("PASO 3: Definiendo funciones de preprocesamiento optimizadas...")

from nltk.corpus import stopwords

# Preprocesamiento paralelo con multiprocessing
def process_batch(texts, functions):
    """Procesa un batch de textos aplicando funciones secuencialmente"""
    for func in functions:
        texts = [func(text) for text in texts]
    return texts

def parallel_preprocess(series, functions, n_workers=None):
    """Preprocesamiento en paralelo usando multiprocessing"""
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)
    
    texts = series.tolist()
    batch_size = max(1, len(texts) // (n_workers * 4))
    batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
    
    with Pool(n_workers) as pool:
        func = partial(process_batch, functions=functions)
        results = pool.map(func, batches)
    
    # Aplanar resultados
    processed = [item for batch in results for item in batch]
    return pd.Series(processed, index=series.index)

# Funciones de limpieza
def clean_text_basic(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[\W_]+', ' ', text)
    return text.strip()

def clean_text_aggressive(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'_+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def remove_stopwords(text):
    doc = nlp(text)
    stop_words = set(stopwords.words('spanish'))
    filtered_tokens = [token.text for token in doc if token.text.lower() not in stop_words and not token.is_punct]
    return ' '.join(filtered_tokens)

def remove_stopwords_strict(text):
    doc = nlp(text)
    stop_words = set(stopwords.words('spanish'))
    filtered_tokens = [token.text for token in doc 
                      if token.text.lower() not in stop_words 
                      and not token.is_punct 
                      and len(token.text) >= 3]
    return ' '.join(filtered_tokens)

def lemmatize_text(text):
    doc = nlp(text)
    lemmas = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return ' '.join(lemmas)

def lemmatize_text_keep_all(text):
    doc = nlp(text)
    lemmas = [token.lemma_ for token in doc]
    return ' '.join(lemmas)

def lemmatize_text_strict(text):
    doc = nlp(text)
    lemmas = [token.lemma_ for token in doc 
             if not token.is_stop 
             and not token.is_punct 
             and len(token.text) >= 3]
    return ' '.join(lemmas)

def remove_accents(text):
    import unicodedata
    text = unicodedata.normalize('NFD', text)
    text = text.encode('ascii', 'ignore').decode('utf-8')
    return text

print("✓ Funciones de preprocesamiento definidas\n")

# ============================================================================
# PASO 4: PIPELINES DE PREPROCESAMIENTO
# ============================================================================

print("PASO 4: Definiendo pipelines de preprocesamiento...")

EXPERIMENT_MODE = 'quick'  # Cambia a 'full' para todos los pipelines

if EXPERIMENT_MODE == 'quick':
    preprocessing_pipelines = {
        'raw': [],
        'clean_basic': [clean_text_basic],
        'clean_standard': [clean_text],
        'clean_aggressive': [clean_text_aggressive],
        'clean_basic_stopwords': [clean_text_basic, remove_stopwords],
        'clean_standard_stopwords': [clean_text, remove_stopwords],
        'clean_aggressive_stopwords': [clean_text_aggressive, remove_stopwords],
        'clean_basic_lemma': [clean_text_basic, lemmatize_text],
        'clean_standard_lemma': [clean_text, lemmatize_text],
        'clean_standard_stopwords_lemma': [clean_text, remove_stopwords, lemmatize_text],
        'clean_aggressive_stopwords_lemma': [clean_text_aggressive, remove_stopwords, lemmatize_text],
        'no_accents_clean_standard_stopwords': [remove_accents, clean_text, remove_stopwords],
    }
else:
    # Full pipeline (43 configuraciones)
    preprocessing_pipelines = {
        'raw': [],
        'clean_basic': [clean_text_basic],
        'clean_standard': [clean_text],
        'clean_aggressive': [clean_text_aggressive],
        'clean_basic_stopwords': [clean_text_basic, remove_stopwords],
        'clean_standard_stopwords': [clean_text, remove_stopwords],
        'clean_aggressive_stopwords': [clean_text_aggressive, remove_stopwords],
        'clean_basic_stopwords_strict': [clean_text_basic, remove_stopwords_strict],
        'clean_standard_stopwords_strict': [clean_text, remove_stopwords_strict],
        'clean_aggressive_stopwords_strict': [clean_text_aggressive, remove_stopwords_strict],
        'clean_basic_lemma': [clean_text_basic, lemmatize_text],
        'clean_standard_lemma': [clean_text, lemmatize_text],
        'clean_aggressive_lemma': [clean_text_aggressive, lemmatize_text],
        'clean_basic_lemma_keepall': [clean_text_basic, lemmatize_text_keep_all],
        'clean_standard_lemma_keepall': [clean_text, lemmatize_text_keep_all],
        'clean_aggressive_lemma_keepall': [clean_text_aggressive, lemmatize_text_keep_all],
        'clean_basic_lemma_strict': [clean_text_basic, lemmatize_text_strict],
        'clean_standard_lemma_strict': [clean_text, lemmatize_text_strict],
        'clean_aggressive_lemma_strict': [clean_text_aggressive, lemmatize_text_strict],
        'clean_basic_stopwords_lemma': [clean_text_basic, remove_stopwords, lemmatize_text],
        'clean_standard_stopwords_lemma': [clean_text, remove_stopwords, lemmatize_text],
        'clean_aggressive_stopwords_lemma': [clean_text_aggressive, remove_stopwords, lemmatize_text],
        'clean_basic_stopwords_strict_lemma_strict': [clean_text_basic, remove_stopwords_strict, lemmatize_text_strict],
        'clean_standard_stopwords_strict_lemma_strict': [clean_text, remove_stopwords_strict, lemmatize_text_strict],
        'clean_aggressive_stopwords_strict_lemma_strict': [clean_text_aggressive, remove_stopwords_strict, lemmatize_text_strict],
        'no_accents_clean_basic': [remove_accents, clean_text_basic],
        'no_accents_clean_standard': [remove_accents, clean_text],
        'no_accents_clean_aggressive': [remove_accents, clean_text_aggressive],
        'no_accents_clean_basic_stopwords': [remove_accents, clean_text_basic, remove_stopwords],
        'no_accents_clean_standard_stopwords': [remove_accents, clean_text, remove_stopwords],
        'no_accents_clean_aggressive_stopwords': [remove_accents, clean_text_aggressive, remove_stopwords],
        'no_accents_clean_basic_lemma': [remove_accents, clean_text_basic, lemmatize_text],
        'no_accents_clean_standard_lemma': [remove_accents, clean_text, lemmatize_text],
        'no_accents_clean_aggressive_lemma': [remove_accents, clean_text_aggressive, lemmatize_text],
        'no_accents_clean_basic_stopwords_lemma': [remove_accents, clean_text_basic, remove_stopwords, lemmatize_text],
        'no_accents_clean_standard_stopwords_lemma': [remove_accents, clean_text, remove_stopwords, lemmatize_text],
        'no_accents_clean_aggressive_stopwords_lemma': [remove_accents, clean_text_aggressive, remove_stopwords, lemmatize_text],
        'stopwords_only': [remove_stopwords],
        'stopwords_strict_only': [remove_stopwords_strict],
        'lemma_only': [lemmatize_text],
        'stopwords_lemma': [remove_stopwords, lemmatize_text],
        'stopwords_strict_lemma_strict': [remove_stopwords_strict, lemmatize_text_strict],
        'no_accents_stopwords_lemma': [remove_accents, remove_stopwords, lemmatize_text],
    }

print(f"✓ Modo: {EXPERIMENT_MODE.upper()}")
print(f"✓ Pipelines: {len(preprocessing_pipelines)}\n")

# ============================================================================
# PASO 5: PREPROCESAMIENTO Y VECTORIZACIÓN (PARALELO)
# ============================================================================

print("PASO 5: Aplicando preprocesamiento y vectorización (Paralelo)...")
print("=" * 80)

vectorization_methods = {
    'binarized': (CountVectorizer, {'binary': True}),
    'frequency': (CountVectorizer, {}),
    'tfidf': (TfidfVectorizer, {})
}

# Guardar etiquetas
with open(os.path.join(VECTORIZED_DIR, 'y_train.pkl'), 'wb') as f:
    pickle.dump(y_train, f)
with open(os.path.join(VECTORIZED_DIR, 'y_test.pkl'), 'wb') as f:
    pickle.dump(y_test, f)

total_combinations = len(preprocessing_pipelines) * len(vectorization_methods)
current = 0

for pipeline_name, functions in preprocessing_pipelines.items():
    print(f"\n{'─' * 80}")
    print(f"Pipeline: {pipeline_name}")
    print(f"{'─' * 80}")
    
    # Preprocesamiento PARALELO
    if functions:
        print("  Aplicando preprocesamiento en paralelo...", end=' ')
        start = datetime.now()
        X_train_processed = parallel_preprocess(X_train, functions, n_workers=n_jobs)
        X_test_processed = parallel_preprocess(X_test, functions, n_workers=n_jobs)
        print(f"✓ ({(datetime.now() - start).total_seconds():.2f}s)")
    else:
        X_train_processed = X_train.copy()
        X_test_processed = X_test.copy()
    
    # Vectorización
    for vect_name, (VectorizerClass, vect_params) in vectorization_methods.items():
        current += 1
        print(f"  [{current}/{total_combinations}] Vectorizando con {vect_name}...", end=' ')
        
        try:
            vectorizer = VectorizerClass(**vect_params)
            X_train_vec = vectorizer.fit_transform(X_train_processed)
            X_test_vec = vectorizer.transform(X_test_processed)
            
            train_file = os.path.join(VECTORIZED_DIR, f'{pipeline_name}_{vect_name}_X_train.pkl')
            test_file = os.path.join(VECTORIZED_DIR, f'{pipeline_name}_{vect_name}_X_test.pkl')
            
            with open(train_file, 'wb') as f:
                pickle.dump(X_train_vec, f)
            with open(test_file, 'wb') as f:
                pickle.dump(X_test_vec, f)
            
            print(f"✓ Shape: {X_train_vec.shape}")
            
        except Exception as e:
            print(f"✗ Error: {str(e)}")

print(f"\n{'=' * 80}")
print("✓ PREPROCESAMIENTO Y VECTORIZACIÓN COMPLETADOS")
print(f"{'=' * 80}\n")

# ============================================================================
# PASO 6: ENTRENAMIENTO CON GPU/CUDA (cuML)
# ============================================================================

print("PASO 6: Entrenando modelos con GPU/CUDA...")
print("=" * 80)

results = []

# Configurar modelos (usar cuML si CUDA disponible)
if cuda_available:
    print("✓ Usando modelos acelerados por GPU (cuML)\n")
    models_config = {
        'Logistic Regression (GPU)': cuLogisticRegression(max_iter=1000),
        'Linear SVC (GPU)': cuSVC(kernel='linear', max_iter=1000),
        'Multinomial NB (GPU)': cuMultinomialNB(),
        'Random Forest (GPU)': cuRandomForestClassifier(n_estimators=100, max_depth=16)
    }
else:
    print("✓ Usando modelos CPU con paralelización\n")
    models_config = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1),
        'Linear SVC': LinearSVC(max_iter=1000, random_state=42),
        'Multinomial NB': MultinomialNB(),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    }

preprocessing_list = list(preprocessing_pipelines.keys())
vectorization_list = list(vectorization_methods.keys())
total_experiments = len(preprocessing_list) * len(vectorization_list) * len(models_config)
exp_count = 0

for prep in preprocessing_list:
    for vec in vectorization_list:
        print(f"\n{'─' * 60}")
        print(f"Datos: {prep} + {vec}")
        print(f"{'─' * 60}")
        
        try:
            with open(os.path.join(VECTORIZED_DIR, f'{prep}_{vec}_X_train.pkl'), 'rb') as f:
                X_train_vec = pickle.load(f)
            with open(os.path.join(VECTORIZED_DIR, f'{prep}_{vec}_X_test.pkl'), 'rb') as f:
                X_test_vec = pickle.load(f)
            with open(os.path.join(VECTORIZED_DIR, 'y_train.pkl'), 'rb') as f:
                y_tr = pickle.load(f)
            with open(os.path.join(VECTORIZED_DIR, 'y_test.pkl'), 'rb') as f:
                y_te = pickle.load(f)
        except FileNotFoundError:
            print(f"  ✗ Archivos no encontrados")
            continue
        
        # Convertir a GPU si está disponible
        if cuda_available:
            # Para cuML necesitamos formato denso o cupy sparse
            if hasattr(X_train_vec, 'toarray'):
                X_train_gpu = cp.array(X_train_vec.toarray())
                X_test_gpu = cp.array(X_test_vec.toarray())
            else:
                X_train_gpu = cp.array(X_train_vec)
                X_test_gpu = cp.array(X_test_vec)
            
            y_tr_gpu = cp.array(y_tr.values if hasattr(y_tr, 'values') else y_tr)
            y_te_gpu = cp.array(y_te.values if hasattr(y_te, 'values') else y_te)
        else:
            X_train_gpu = X_train_vec
            X_test_gpu = X_test_vec
            y_tr_gpu = y_tr
            y_te_gpu = y_te
        
        for model_name, model in models_config.items():
            exp_count += 1
            print(f"  [{exp_count}/{total_experiments}] {model_name}...", end=' ')
            
            try:
                start_time = datetime.now()
                model.fit(X_train_gpu, y_tr_gpu)
                train_time = (datetime.now() - start_time).total_seconds()
                
                y_pred = model.predict(X_test_gpu)
                
                # Convertir predicciones de GPU a CPU si es necesario
                if cuda_available:
                    y_pred_cpu = cp.asnumpy(y_pred)
                    y_te_cpu = cp.asnumpy(y_te_gpu)
                else:
                    y_pred_cpu = y_pred
                    y_te_cpu = y_te_gpu
                
                metrics = {
                    'model': model_name,
                    'preprocessing': prep,
                    'vectorization': vec,
                    'accuracy': accuracy_score(y_te_cpu, y_pred_cpu),
                    'precision_macro': precision_score(y_te_cpu, y_pred_cpu, average='macro', zero_division=0),
                    'recall_macro': recall_score(y_te_cpu, y_pred_cpu, average='macro', zero_division=0),
                    'f1_macro': f1_score(y_te_cpu, y_pred_cpu, average='macro', zero_division=0),
                    'precision_weighted': precision_score(y_te_cpu, y_pred_cpu, average='weighted', zero_division=0),
                    'recall_weighted': recall_score(y_te_cpu, y_pred_cpu, average='weighted', zero_division=0),
                    'f1_weighted': f1_score(y_te_cpu, y_pred_cpu, average='weighted', zero_division=0),
                    'training_time': train_time
                }
                results.append(metrics)
                
                print(f"Acc: {metrics['accuracy']:.4f}, F1: {metrics['f1_weighted']:.4f}, Time: {train_time:.2f}s")
                
            except Exception as e:
                print(f"✗ Error: {str(e)}")

print(f"\n{'=' * 80}")
print("✓ ENTRENAMIENTO COMPLETADO")
print(f"{'=' * 80}\n")

# ============================================================================
# PASO 7: RESULTADOS Y VISUALIZACIONES
# ============================================================================

print("PASO 7: Generando resultados...")

results_df = pd.DataFrame(results)

csv_path = os.path.join(RESULTS_DIR, 'model_results_gpu.csv')
results_df.to_csv(csv_path, index=False)
print(f"✓ Resultados: {csv_path}")

# Visualizaciones
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Comparación de Modelos (GPU Accelerated)', fontsize=16, fontweight='bold')

pivot_acc = results_df.pivot_table(values='accuracy', index='model', columns='vectorization', aggfunc='mean')
sns.heatmap(pivot_acc, annot=True, fmt='.3f', cmap='YlGnBu', ax=axes[0, 0])
axes[0, 0].set_title('Accuracy Promedio')

prep_f1 = results_df.groupby(['preprocessing', 'model'])['f1_weighted'].mean().reset_index()
top_preps = prep_f1.groupby('preprocessing')['f1_weighted'].mean().nlargest(10).index
prep_f1_filtered = prep_f1[prep_f1['preprocessing'].isin(top_preps)]
sns.barplot(data=prep_f1_filtered, x='preprocessing', y='f1_weighted', hue='model', ax=axes[0, 1])
axes[0, 1].set_title('F1-Score Top 10')
axes[0, 1].tick_params(axis='x', rotation=45)

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
axes[1, 0].set_title(f'Mejor: {best["model"]}\n{best["preprocessing"]} + {best["vectorization"]}')
axes[1, 0].legend()

top_10 = results_df.nlargest(10, 'f1_weighted')
top_10['config'] = top_10.apply(lambda x: f"{x['model'][:12]}\n{x['preprocessing'][:15]}", axis=1)
axes[1, 1].barh(range(len(top_10)), top_10['f1_weighted'])
axes[1, 1].set_yticks(range(len(top_10)))
axes[1, 1].set_yticklabels(top_10['config'], fontsize=8)
axes[1, 1].set_title('Top 10 Configuraciones')
axes[1, 1].invert_yaxis()

plt.tight_layout()
plot_path = os.path.join(RESULTS_DIR, 'model_comparison_gpu.png')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"✓ Gráfico: {plot_path}")

# Reporte final
print("\n" + "=" * 80)
print("REPORTE FINAL")
print("=" * 80)
print(f"\nMEJOR CONFIGURACIÓN:")
print(f"  Modelo: {best['model']}")
print(f"  Preprocesamiento: {best['preprocessing']}")
print(f"  Vectorización: {best['vectorization']}")
print(f"  Accuracy: {best['accuracy']:.4f}")
print(f"  F1-Score: {best['f1_weighted']:.4f}")
print(f"  Tiempo: {best['training_time']:.2f}s")

print(f"\nTIEMPO TOTAL POR MODELO:")
for model in results_df['model'].unique():
    total_time = results_df[results_df['model'] == model]['training_time'].sum()
    avg_time = results_df[results_df['model'] == model]['training_time'].mean()
    best_f1 = results_df[results_df['model'] == model]['f1_weighted'].max()
    print(f"  {model}: Total={total_time:.2f}s, Avg={avg_time:.2f}s, Best F1={best_f1:.4f}")

print("\n" + "=" * 80)
print("✓ ANÁLISIS COMPLETO FINALIZADO")
print("=" * 80)
