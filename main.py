"""
Biblical Text Mining - Pipeline Principal
Curso: Programación Científica | UCN 2026
Nombres de las intengrantes: 
Paulette Bauer - 215743853-3
Valentina Vergara - 21833059-2
María Salas - 21591568-9
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from collections import Counter

# Asegurar que el directorio raíz esté en el path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.preprocessor import TextPreprocessor, TFIDFVectorizer
from src.search_engine import SemanticSearchEngine
from src.classifier import VerseClassifier
from src.text_generator import TextGenerator, NGramModel
from src.sentiment import SentimentAnalyzer
from src.visualizer import Visualizer


# ══════════════════════════════════════════════
# CONFIGURACIÓN
# ══════════════════════════════════════════════
DATA_PATH   = "data/bible.csv"
OUTPUT_DIR  = "outputs"
TFIDF_FEATURES = 5000
TFIDF_MIN_DF   = 3
CUSTOM_N       = 4    # n para el modelo n-gram personalizado
SEARCH_SEED    = "In the beginning God"
TOP_K_SEARCH   = 5
RANDOM_SEED    = 42

np.random.seed(RANDOM_SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def separator(title: str) -> None:
    print(f"\n{'═'*60}")
    print(f"  {title}")
    print('═'*60)


def main():
    # ──────────────────────────────────────────
    # 0. Carga de datos
    # ──────────────────────────────────────────
    separator("0. CARGA DE DATOS")
    df = pd.read_csv(DATA_PATH)
    print(f"Total versículos: {len(df)}")
    print(f"Libros: {df['book'].nunique()}")
    print(f"Testamentos: {df['testament'].value_counts().to_dict()}")

    # ──────────────────────────────────────────
    # 1. PREPROCESAMIENTO
    # ──────────────────────────────────────────
    separator("1. PREPROCESAMIENTO")

    # Extra stopwords bíblicas (palabras muy frecuentes sin valor semántico en este dominio)
    extra_sw = ['thou', 'thee', 'thy', 'ye', 'hath', 'shalt', 'hast', 'unto',
                'shall', 'said', 'came', 'come', 'forth', 'thus', 'also',
                'upon', 'saith', 'thereof', 'thine', 'doth', 'therefore', 'wherefore']

    preprocessor = TextPreprocessor(language='english', extra_stopwords=extra_sw)
    print("Construyendo vocabulario...")
    vocab = preprocessor.build_vocabulary_from_series(df['text'])
    print(f"Vocabulario: {len(vocab)} términos únicos")
    print(f"Top 10 palabras: {preprocessor.get_top_words(10)}")

    print("Tokenizando corpus...")
    tokenized_corpus = df['text'].apply(preprocessor.preprocess).tolist()
    df['tokens'] = tokenized_corpus

    # Word frequencies por testamento
    ot_mask = df['testament'] == 'Old Testament'
    nt_mask = df['testament'] == 'New Testament'
    ot_tokens = [t for tokens in df[ot_mask]['tokens'] for t in tokens]
    nt_tokens = [t for tokens in df[nt_mask]['tokens'] for t in tokens]
    freq_ot = Counter(ot_tokens)
    freq_nt = Counter(nt_tokens)

    # ──────────────────────────────────────────
    # 2. TF-IDF
    # ──────────────────────────────────────────
    separator("2. VECTORIZACIÓN TF-IDF")
    vectorizer = TFIDFVectorizer(max_features=TFIDF_FEATURES, min_df=TFIDF_MIN_DF)
    print("Calculando TF-IDF...")
    tfidf_matrix = vectorizer.fit_transform(tokenized_corpus)
    print(f"Matriz TF-IDF: {tfidf_matrix.shape} | Vocabulario: {len(vectorizer.vocabulary)} términos")

    # Centroides por libro
    books = df['book'].tolist()
    unique_books = df['book'].unique().tolist()
    book_centroids = {}
    for book in unique_books:
        idx = [i for i, b in enumerate(books) if b == book]
        book_centroids[book] = tfidf_matrix[idx].mean(axis=0)

    # ──────────────────────────────────────────
    # 3. VISUALIZACIONES
    # ──────────────────────────────────────────
    separator("3. VISUALIZACIONES")
    viz = Visualizer(output_dir=OUTPUT_DIR)

    print("  [1/8] Distribución de longitud de versículos...")
    viz.plot_verse_length_distribution(df)

    print("  [2/8] Versículos por libro...")
    viz.plot_verses_per_book(df)

    print("  [3/8] Top palabras más frecuentes...")
    viz.plot_top_words(preprocessor.word_frequencies, top_n=30)

    print("  [4/8] Heatmap de similitud entre libros (OBLIGATORIO)...")
    viz.plot_book_similarity_heatmap(df, tfidf_matrix)

    print("  [5/8] PCA de versículos...")
    viz.plot_pca_verses(df, tfidf_matrix, max_verses=6000)

    print("  [6/8] Nube de palabras...")
    viz.plot_wordcloud(freq_ot, freq_nt)

    # ──────────────────────────────────────────
    # 4. MOTOR DE BÚSQUEDA SEMÁNTICO
    # ──────────────────────────────────────────
    separator("4. MOTOR DE BÚSQUEDA SEMÁNTICO")
    engine = SemanticSearchEngine(preprocessor, vectorizer)
    engine.index(df, tfidf_matrix)

    queries = [
        "In the beginning God created the heaven",
        "love thy neighbor as thyself",
        "blessed are the poor in spirit",
    ]

    all_search_results = {}
    for query in queries:
        print(f"\n  Query: '{query}'")
        results = engine.search(query, k=TOP_K_SEARCH)
        all_search_results[query] = results
        print(results[['rank', 'book', 'chapter', 'verse', 'similarity']].to_string(index=False))

    # ──────────────────────────────────────────
    # 5. CLASIFICADOR DE VERSÍCULOS
    # ──────────────────────────────────────────
    separator("5. CLASIFICADOR DE VERSÍCULOS")
    classifier = VerseClassifier(preprocessor, vectorizer)
    classifier.prepare_data(df, tfidf_matrix)
    results_lr = classifier.train_logistic()

    print(f"\n  Accuracy Regresión Logística: {results_lr['accuracy']:.4f} ({results_lr['accuracy']*100:.2f}%)")

    # Guardar métricas
    metrics_df = pd.DataFrame(results_lr['report']).T
    metrics_df.to_csv(f"{OUTPUT_DIR}/classification_report.csv")

    # Visualización de matriz de confusión
    labels = [classifier.label_decoder[i] for i in range(len(classifier.label_decoder))]
    print("  [7/8] Matriz de confusión...")
    viz.plot_confusion_matrix(results_lr['confusion_matrix'], labels, top_n=20)

    # Ejemplo de predicción
    test_verse = "Jesus wept."
    pred = classifier.predict_verse(test_verse, model='logistic')
    print(f"\n  Predicción para '{test_verse}':")
    print(f"    Libro predicho: {pred['predicted_book']} (confianza: {pred['confidence']:.3f})")
    print(f"    Top 3: {pred['top3']}")

    # ──────────────────────────────────────────
    # 6. MODELOS GENERATIVOS N-GRAM
    # ──────────────────────────────────────────
    separator("6. MODELOS GENERATIVOS N-GRAM")
    generator = TextGenerator()
    generator.train_all(tokenized_corpus, custom_n=CUSTOM_N)

    seed_word = "god"
    print(f"\n  Textos generados con seed='{seed_word}':")
    generated_texts = generator.generate_with_all(seed_word=seed_word, max_length=20, temperature=0.8)
    for model_name, text in generated_texts.items():
        print(f"    [{model_name:8s}] {text}")

    # Calcular perplejidad (muestra del 5% para velocidad)
    print("\n  Calculando perplejidad...")
    n_test = max(100, len(tokenized_corpus) // 20)
    test_idx = np.random.choice(len(tokenized_corpus), n_test, replace=False)
    test_corpus = [tokenized_corpus[i] for i in test_idx]

    perplexities = {}
    for name, model in generator.models.items():
        try:
            ppl = model.perplexity(test_corpus)
            perplexities[name] = min(ppl, 9999)  # cap para visualización
            print(f"    {name:8s}: perplejidad = {perplexities[name]:.2f}")
        except Exception as e:
            print(f"    {name:8s}: error al calcular perplejidad ({e})")
            perplexities[name] = 9999

    print("  [8/8] Comparación de modelos n-gram...")
    viz.plot_ngram_comparison(perplexities, generated_texts)

    # ──────────────────────────────────────────
    # 7. ANÁLISIS DE SENTIMIENTO
    # ──────────────────────────────────────────
    separator("7. ANÁLISIS DE SENTIMIENTO")
    sentiment_analyzer = SentimentAnalyzer()

    print("  Calculando sentimientos (puede tardar unos segundos)...")
    df_sentiment = sentiment_analyzer.analyze_corpus(df)

    book_sent = sentiment_analyzer.aggregate_by_book(df_sentiment)
    extremes  = sentiment_analyzer.get_extreme_books(book_sent, top_n=5)

    print("\n  Top 5 libros MÁS POSITIVOS:")
    print(extremes['most_positive'].to_string(index=False))

    print("\n  Top 5 libros MÁS NEGATIVOS:")
    print(extremes['most_negative'].to_string(index=False))

    viz.plot_sentiment_by_book(book_sent)
    viz.plot_sentiment_evolution(df_sentiment)

    # Guardar resultado de sentimiento
    book_sent.to_csv(f"{OUTPUT_DIR}/sentiment_by_book.csv", index=False)

    # ──────────────────────────────────────────
    # 8. RESUMEN FINAL
    # ──────────────────────────────────────────
    separator("8. RESUMEN FINAL")
    print(f"""
  Dataset:
    - Versículos totales   : {len(df):,}
    - Libros               : {df['book'].nunique()}
    - AT / NT              : {df[ot_mask].shape[0]:,} / {df[nt_mask].shape[0]:,}

  Preprocesamiento:
    - Vocabulario          : {len(vocab):,} tokens únicos
    - TF-IDF features      : {tfidf_matrix.shape[1]:,}

  Clasificador (Regresión Logística):
    - Accuracy             : {results_lr['accuracy']*100:.2f}%

  Modelos N-gram (perplejidad ↓ mejor):
    {chr(10).join(f'    {k:8s}: {v:.2f}' for k, v in perplexities.items())}

  Visualizaciones guardadas en /{OUTPUT_DIR}/
  """)

    print("  ¡Pipeline completado exitosamente!")
    return {
        'df': df,
        'df_sentiment': df_sentiment,
        'tfidf_matrix': tfidf_matrix,
        'vectorizer': vectorizer,
        'preprocessor': preprocessor,
        'classifier': classifier,
        'engine': engine,
        'generator': generator,
        'book_sent': book_sent,
        'perplexities': perplexities,
        'generated_texts': generated_texts,
        'all_search_results': all_search_results,
        'freq_ot': freq_ot,
        'freq_nt': freq_nt,
    }


if __name__ == "__main__":
    results = main()
