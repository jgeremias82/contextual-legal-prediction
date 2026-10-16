# Jupyter Notebooks — Experimental Pipeline

This directory contains the experimental notebooks implementing the end-to-end framework for Legal Judgment Prediction (LJP) on Brazilian Habeas Corpus proceedings.

> [!IMPORTANT]
> **Guidance for Data Scientists**:
> - **Step-by-Step Execution & Parameterization**: The notebooks in this directory feature extensive parameterization (`DATASET_TYPE`, `DATASET_VERSION`, `SELECTED_MODEL_ID`, `APPROACHES`, `STRIDE`, `NROWS_LOADING`, etc.) that allows tailoring the pipeline for different experimental objectives, hardware configurations, and diagnostic runs. It is strongly recommended to read through the documentation of each notebook and execute the cells **step-by-step**, ensuring that settings, dependencies, and execution modes (`TRAIN_MODE` vs. `INFERENCE_MODE`) are verified prior to full execution.
> - **Environment & Path Re-establishment**: The default paths for reading and writing data sources (such as `base_path`, `BASE_PATH_DATASET`, and `BASE_MODEL_SAVED`) as well as loading external preprocessing functions (via `%run`) are configured for execution in **Google Colab** with Google Drive mounted (`/content/drive/MyDrive/...`). Prior to running the notebooks, these directory paths and file locations must be carefully re-established and adjusted to match your target environment or local directory structure.

---

## Notebook Overview & Workflow

The notebooks are organized sequentially according to the project pipeline:

```text
Preprocess_DataSet.ipynb
       │
       ├── imports functions from ──► preprocessing.ipynb
       ▼
EDA_DataSet(Tokenization_Undersampling_and_Tradeoff).ipynb
       ▼
BERT_HabeasCorpus(Train_and_Inference_Models).ipynb
```

---

## Detailed Notebook Descriptions

### 1. `Preprocess_DataSet.ipynb`
*Data Consolidation, Metadata Extraction, and Label Structuring*

- **Purpose**: Consolidates raw public data harvested from the Open Data Portal and builds the unified tabular dataset for the research.
- **Key Functionality**:
  - Integrates heterogeneous annual records (2021 to 2025) across monocratic decisions (*decisões monocráticas*) and collegiate court judgments (*acórdãos*).
  - Extracts and structures judicial metadata, including process class, registration number (`NUMEROREGISTRO`), rapporteur (`MINISTRO_RELATOR`), court organ, judgment date, and publication date.
  - Isolates the judicial report (*relatório*) and the full ruling text (*teor*).
  - Maps judicial outcomes into binary target labels (`DECISAO_FAVORAVEL`: `S` / `N` or 1 / 0) based on standardized judgment description dictionaries (granted vs. denied petitions).
  - Handles missing data, removes duplicates, and filters for Habeas Corpus records (`CLASSE = 'HC'`).
- **Configurable Parameters**:
  - `NROWS`: Enables loading a small fraction of records for rapid debugging (`None` for complete processing).
  - `CLASSE`: Target procedural class (e.g., `'HC'`, `'AREsp'`, `'REsp'`).
  - `TIPO_CONSTRUCAO` & `LISTA_TIPO_ARQUIVO`: Specifies document categories to process (`'geral'`, `['decisoes', 'acordaos']`).
  - `ANO_INI`, `ANO_FIM`: Year boundaries for dataset collection (2021–2025).
  - `base_path`: Base storage directory on Google Drive.

---

### 2. `preprocessing.ipynb`
*Legal NLP Normalization & Linguistic Structure Preservation*

- **Purpose**: Provides specialized textual preprocessing routines tailored specifically for Transformer-based architectures operating on Brazilian legal texts.
- **Key Functionality**:
  - **Noise Removal**: Eliminates HTML tags and formatting artifacts via regex (`remove_noise`).
  - **Spacing & Typographic Normalization**: Fixes irregular whitespace, standardizes punctuation spacing, and ensures proper sentence boundary spacing (`remove_whitespace`).
  - **Legal Acronym & Abbreviation Protection**: Employs placeholder mechanisms (`_protect_dots`, `_restore_dots`) to prevent improper sentence splitting on common legal abbreviations (`art.`, `c.c.`, `nº`, `fls.`, `min.`, `rel.`) and numeric legal notations (e.g., `1.125,00`).
  - **Morphosyntactic Structure Preservation**: Justifies and enforces the retention of critical grammatical elements for legal semantics (negation adverbs like *não*, *nem*, *sem*; adversative conjunctions like *mas*, *porém*; and pronominal enclisis/mesoclisis like *condenou-se*, *dar-se-á*) to preserve sentence logic and bidirectional attention in BERT.
  - Main pipeline entry point: `pre_processamento(texto, remover_stopwords, ...)`.

---

### 3. `EDA_DataSet(Tokenization_Undersampling_and_Tradeoff).ipynb`
*Exploratory Analysis, Token Length Profiling, Class Balancing & Dataset Partitioning*

- **Purpose**: Analyzes textual and class distributions, defines document length thresholds, balances classes via undersampling, and produces the official train/test splits.
- **Key Functionality**:
  - **Exploratory Data Analysis**: Generates publication-ready stacked visualizations of proceeding volumes, class ratios over time, and outcome distributions.
  - **Tokenization Profiling**: Measures token counts in the `RELATORIO` field using BERT tokenizers, identifying long-document frontiers exceeding the 512-token limit.
  - **Empirical Thresholding**: Separates short legal documents ($\le 510$ tokens) from long documents ($> 510$ tokens, up to several thousand tokens), strictly reserving 2 positions for the `[CLS]` and `[SEP]` special tokens within the model's 512-token maximum capacity.
  - **Class Undersampling & Outlier Filtering**: Mitigates severe real-world class imbalance (predominance of denied petitions) by undersampling majority classes, while filtering extreme token length outliers.
  - **Stratified Partitioning**: Generates reproducible, stratified train/test partitions (e.g., 70/30 in v4, 80/20 in v5) saved as separate versioned CSV files.
  - **Inference Trade-off Dataset Preparation**: Samples dedicated subsets for subsequent inference latency benchmarking.
- **Configurable Parameters**:
  - `RANDOM_SEED`: Fixed seed (42) for reproducible data sampling.
  - `TEST_SIZE_SPLIT` / `SPLIT_TEST_SIZE`: Test split ratio (e.g., 0.3 or 0.2).
  - `DATASET_TYPE`: Target dataset segment (`'LONG'`, `'SMALL'`, or `'MIX'`).
  - `DATASET_VERSION`: Version identifier for exported files (`v4`, `v5`).
  - `SAVE_GRAPHIC`, `DPI_RESOLUTION`, `FONT_SIZE`: Settings for rendering high-resolution scientific figures (IEEE / ACM standard).

---

### 4. `BERT_HabeasCorpus(Train_and_Inference_Models).ipynb`
*Model Training, Sliding-Window Aggregation, Evaluation & Latency Trade-offs*

- **Purpose**: Executes model training, representation aggregation (sliding-window + pooling), performance evaluation, and inference latency profiling across BERT-based models.
- **Key Functionality**:
  - **Multiple Model Architectures**: Supports benchmarking across domain-specific and general-purpose backbones:
    - Legal BERT
    - Legal-BERTimbau
    - BERTimbau Base
    - Multilingual BERT
  - **Sliding-Window & Context Preservation**:
    - Chunks long legal documents exceeding 512 tokens with configurable stride and overlap (e.g., Stride 0 / contiguous chunks, Stride 128, Stride 256, Stride 384).
  - **Representation Aggregation**:
    - Truncation baseline (first 512 tokens).
    - Mean Pooling aggregation across chunk representations.
    - Max Pooling aggregation across chunk representations.
    - Classification head on top of frozen backbones or fine-tuning.
  - **Diagnostic Tools**:
    - Verifies chunk generation, layer freeze status, and DataLoader throughput before launching large-scale training.
  - **Metrics & Trade-off Analysis**:
    - Calculates Accuracy, Precision, Recall, Macro F1, Weighted F1, and ROC-AUC curves.
    - Profiles inference latency (milliseconds per document) and computational cost across models and aggregation strategies.
- **Configurable Parameters**:
  - `TRAIN_MODE` / `INFERENCE_MODE`: Toggles training from scratch vs. loading existing checkpoints for inference and evaluation.
  - `FORCE_RETRAIN`: Overwrites existing checkpoints when set to `True`.
  - `SELECTED_MODEL_ID`: Chooses the active model checkpoint configuration.
  - `DATASET_TYPE`: Selects dataset scale (`'LONG'`, `'SMALL'`, `'MIX'`).
  - `DATASET_VERSION`: Dataset version to read (e.g., `4`).
  - `CHUNK_SIZE` & `EFFECTIVE_CHUNK_TOKENS`: Maximum chunk length (512) and effective text capacity excluding special tokens `[CLS]`/`[SEP]`.
  - `APPROACHES`: Dictionary enabling/disabling specific aggregation strategies (Truncation, Mean Pooling, Max Pooling with varied strides).
  - `BATCH_SIZE`, `EPOCHS`, `NROWS_LOADING`: Execution hyperparameters and optional sample size limits.
  - `DIAGNOSIS_MODE`: Runs sanity checks on tensors, embeddings, and chunks.
