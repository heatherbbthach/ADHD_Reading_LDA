# Topic Modeling of ADHD Discourse on Reading Challenges

Code accompanying:

Thach, H., & Ridenour, L. (2025). Topic modeling of ADHD discourse on reading challenges: Cognitive differences and systemic factors. *Neurodiversity, 3*, 1–16. https://doi.org/10.1177/27546330251395890

## Overview

This repository contains the Python code used for the computational analysis reported in the article. The study used Latent Dirichlet Allocation (LDA) to analyze 6,295 Reddit posts and comments concerning ADHD and reading comprehension.

The repository includes code for:

- Collecting Reddit posts and comments using the Python Reddit API Wrapper (PRAW).
- Cleaning and preprocessing text using regular expressions, tokenization, and stopword removal.
- Evaluating candidate LDA models using coherence measures.
- Fitting a final three-topic LDA model.
- Generating topic visualizations using pyLDAvis.
- Identifying documents with high topic probabilities for qualitative interpretation.

The computational analysis was complemented by qualitative interpretation of the topics to examine cognitive differences and systemic factors shaping reading challenges among individuals with ADHD.

## Repository Structure

```text
ADHD_Reading_LDA/
├── README.md
├── CITATION.cff
├── requirements.txt
├── .gitignore
├── src/
│   ├── 01_collect_reddit_data.py
│   └── 02_topic_modeling.py
├── data/
│   └── README.md
└── outputs/
    └── README.md
```

## Requirements

- Python 3.11
- Dependencies listed in `requirements.txt`
- Reddit API credentials (required only for data collection)

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/heatherbbthach/ADHD_Reading_LDA.git
   cd ADHD_Reading_LDA
   ```

2. Create a virtual environment using Python 3.11:

   ```bash
   python3.11 -m venv .venv
   ```

3. Activate the virtual environment.

   **macOS/Linux:**

   ```bash
   source .venv/bin/activate
   ```

   **Windows (PowerShell):**

   ```powershell
   .venv\Scripts\Activate.ps1
   ```

4. Install the dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

All scripts should be executed from the repository's root directory so that relative file paths resolve correctly.

## Usage

The workflow consists of two parts: Reddit data collection and topic modeling.

### Part 1: Reddit Data Collection

**Script:** `src/01_collect_reddit_data.py`

This script uses PRAW to retrieve Reddit submissions matching the search query `"ADHD" "Reading Comprehension"` across all subreddits. Search results are sorted by relevance (the PRAW default), with a limit of 500 submissions. Associated comments are also collected.

Before executing the script, create a file named `credentials.csv` in the repository's root directory containing the following column headers:

```text
client_id,client_secret,password,user_agent,username
```

Populate the file with credentials associated with your Reddit API access.

The credentials file is excluded from version control through `.gitignore` and should never be uploaded to GitHub.

Run the collection script:

```bash
python src/01_collect_reddit_data.py
```

The resulting dataset is written to:

```text
data/reddit_adhd_data.csv
```

The CSV contains six columns: `ID`, `Parent ID`, `Post Title`, `Text`, `Post URL`, and `Type`.

**Note:** The dataset collected through this script will vary depending on when it is executed, Reddit content availability, search indexing, and API access. Running the script does not reproduce the exact corpus of 6,295 records analyzed in the published article.

### Part 2: Topic Modeling

**Script:** `src/02_topic_modeling.py`

This script processes the collected Reddit data and performs the LDA analysis.

The analysis includes:

1. Reading the CSV dataset and selecting the relevant text fields.
2. Cleaning text using regular expressions.
3. Generating a word cloud for exploratory analysis.
4. Tokenizing text and removing predefined and custom stopwords.
5. Evaluating models containing 1–15 topics using C_V, U_Mass, C_UCI, and C_NPMI coherence measures.
6. Fitting a final three-topic LDA model using 50 training passes.
7. Calculating final model coherence values.
8. Generating interactive topic visualizations using pyLDAvis.
9. Identifying documents with high probabilities of belonging to particular topics.

To run the analysis, ensure that `data/reddit_adhd_data.csv` exists, then execute:

```bash
python src/02_topic_modeling.py
```

The script generates coherence calculations, topic-modeling results, and visualizations. Generated files are saved to the repository root when the script is executed as instructed. See `outputs/README.md` for additional information.

### Reproducibility Considerations

The scripts preserve the analytical procedures used in the published study, with minor adjustments to support execution outside Google Colab.

Because the original Reddit dataset is not distributed, running the workflow with newly collected data will reproduce the analytical procedure rather than the exact published findings. Results may vary depending on the content of the dataset retrieved.

## Data Availability

Raw Reddit posts and comments are not distributed with this repository to protect user privacy.

The published study used publicly accessible Reddit discussions; however, the researchers took precautions to minimize potential identification of individuals whose posts were analyzed.

The data-collection script is provided to document the original collection procedure and support methodological transparency.

Additional information about the dataset structure is available in `data/README.md`.

## Citation

If you use this code in your research, please cite the accompanying publication:

Thach, H., & Ridenour, L. (2025). Topic modeling of ADHD discourse on reading challenges: Cognitive differences and systemic factors. *Neurodiversity, 3*, 1–16. https://doi.org/10.1177/27546330251395890

Citation metadata is also provided in [`CITATION.cff`](CITATION.cff).
