# 📖 Biblical Text Mining
### Laboratorio 2 — Programación Científica | UCN 2026
**Profesor:** Cristhian A. Rabi R.

---

## Descripción

Sistema de análisis computacional del corpus bíblico (King James Bible, 31.100 versículos, 66 libros) aplicando técnicas de NLP, representación vectorial TF-IDF, clasificación automática, modelos generativos de lenguaje y análisis de sentimiento.

---

## Estructura del Proyecto

```
biblical_text_mining/
├── data/
│   ├── en_kjv.json          # Dataset original (King James Bible)
│   └── bible.csv            # Dataset procesado en formato plano
├── src/
│   ├── __init__.py
│   ├── preprocessor.py      # TextPreprocessor + TFIDFVectorizer (implementado manualmente)
│   ├── search_engine.py     # CosineSimilarity + SemanticSearchEngine
│   ├── classifier.py        # VerseClassifier (Regresión Logística + Naive Bayes)
│   ├── text_generator.py    # NGramModel + TextGenerator (unigram/bigram/trigram/ngram)
│   ├── sentiment.py         # SentimentAnalyzer (VADER)
│   └── visualizer.py        # Visualizer (10 visualizaciones)
├── outputs/                 # Gráficos generados
├── main.py                  # Pipeline principal
└── README.md
```

---

## Instalación

```bash
# Clonar repositorio
git clone <url-del-repo>
cd biblical_text_mining

# Instalar dependencias
pip install pandas numpy matplotlib seaborn scikit-learn nltk networkx wordcloud textblob

# Descargar datos NLTK necesarios
python3 -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('vader_lexicon')"

# Ejecutar pipeline completo
python3 main.py
```

---

## Módulos y Clases

### `src/preprocessor.py`
| Clase | Responsabilidad |
|---|---|
| `TextPreprocessor` | Limpieza, tokenización, eliminación de stopwords, vocabulario |
| `TFIDFVectorizer` | **Implementación manual** de TF-IDF (sin sklearn). Calcula TF relativo, IDF suavizado y normalización L2 |

**Pipeline de preprocesamiento:**
1. Conversión a minúsculas
2. Eliminación de anotaciones `{...}` (propias de KJV)
3. Eliminación de puntuación, números y caracteres especiales
4. Tokenización por espacio
5. Eliminación de stopwords (NLTK + stopwords bíblicas específicas)
6. Construcción de vocabulario ordenado por frecuencia

### `src/search_engine.py`
| Clase | Responsabilidad |
|---|---|
| `CosineSimilarity` | Cálculo manual de similitud del coseno (puntual y batch) |
| `SemanticSearchEngine` | Motor de búsqueda: indexa corpus TF-IDF y encuentra los K versículos más similares |

### `src/classifier.py`
| Clase | Responsabilidad |
|---|---|
| `VerseClassifier` | Clasificación multiclase: dado un versículo, predice el libro bíblico |

**Modelos entrenados:**
- Regresión Logística (sklearn, solver lbfgs)
- Accuracy obtenida: ~47% sobre 56 libros (problema difícil: estilo muy similar entre libros)

### `src/text_generator.py`
| Clase | Responsabilidad |
|---|---|
| `NGramModel` | Modelo estadístico de lenguaje con tokens `<START>` / `<END>`, backoff y muestreo por temperatura |
| `TextGenerator` | Orquestador: entrena y compara modelos unigram, bigram, trigram y 4-gram |

### `src/sentiment.py`
| Clase | Responsabilidad |
|---|---|
| `SentimentAnalyzer` | Analiza sentimiento con VADER; agrega por libro y capítulo; identifica extremos |

### `src/visualizer.py`
| Clase | Responsabilidad |
|---|---|
| `Visualizer` | Genera las 10 visualizaciones del laboratorio |

---

## Visualizaciones generadas

| Archivo | Descripción |
|---|---|
| `01_verse_length_distribution.png` | Histograma de longitud de versículos (AT vs NT) |
| `02_verses_per_book.png` | Cantidad de versículos por libro |
| `03_top_words.png` | Top 30 palabras más frecuentes del corpus |
| `04_book_similarity_heatmap.png` | **Heatmap 66×66** de similitud del coseno entre libros _(obligatorio)_ |
| `05_pca_verses.png` | Proyección PCA 2D de 6.000 versículos (coloreado por testamento) |
| `06_sentiment_by_book.png` | Sentimiento promedio VADER por libro |
| `07_sentiment_evolution.png` | Evolución del sentimiento capítulo a capítulo |
| `08_ngram_perplexity.png` | Comparación de perplejidad por modelo (unigram→4gram) |
| `09_confusion_matrix.png` | Matriz de confusión del clasificador (Top 20 libros) |
| `10_wordclouds.png` | Nubes de palabras AT y NT |

---

## Decisiones de Diseño

### Stopwords bíblicas adicionales
Se añadieron palabras propias del inglés arcaico de la KJV (`thou`, `thee`, `thy`, `hath`, `shalt`, `unto`…) que son extremadamente frecuentes pero no aportan contenido semántico relevante para las tareas de similitud y clasificación.

### TF-IDF manual
`TFIDFVectorizer` implementa la fórmula estándar:
- `TF(t,d) = count(t,d) / |d|`
- `IDF(t) = log((1+N)/(1+df(t))) + 1` (suavizado para evitar división por cero)
- Normalización L2 por fila

### Similitud del coseno manual
`CosineSimilarity.compute_batch` realiza el producto punto entre el vector query normalizado y la matriz de documentos (ya normalizados en el paso TF-IDF), equivalente a `q @ M.T` en una sola operación vectorizada.

### Clasificación: limitaciones esperadas
Con 56 libros y estilos tan variados, una accuracy de ~47% es razonable. Los libros con más versículos (Psalms, Isaiah, Jeremiah) se predicen bien; libros cortos (Obadiah, 3 John) tienen menor representación en el entrenamiento.

### Modelos N-gram: comportamiento esperado
- **Unigram**: genera palabras independientes; baja coherencia pero alta diversidad léxica
- **Bigram**: introduce contexto local de 1 palabra; coherencia mejora levemente
- **Trigram/4-gram**: contexto más largo; secuencias más "bíblicas" pero mayor riesgo de reproducir fragmentos del corpus

### Análisis de sentimiento: limitaciones
VADER fue diseñado para redes sociales modernas, no para texto bíblico del siglo XVII. El inglés arcaico (KJV) contiene vocabulario que VADER puede malinterpretar. Los resultados son indicativos, no definitivos.

---

## Requisitos del sistema

- Python 3.9+
- pandas, numpy, matplotlib, seaborn
- scikit-learn (solo para PCA y Regresión Logística)
- nltk (stopwords + VADER)
- wordcloud (opcional, para nubes de palabras)

---

## Autores

Equipo — Ingeniería Civil en Computación e Informática, UCN 2026
