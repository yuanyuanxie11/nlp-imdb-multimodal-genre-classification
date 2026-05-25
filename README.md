# NLP IMDB Multimodal Genre Classification

This repository is our team workspace for the Northwestern Text Analytics final project. We are building a movie genre classification system that uses plot summaries as the core signal and leaves room for a bonus multimodal extension using poster images.

Our goal is not just to complete the assignment. We want to build something polished, interpretable, collaborative, and demo-friendly enough to stand out in both technical quality and presentation.

## Project Vision

We will investigate how well different NLP and deep learning methods can predict whether a movie belongs to:

- Action
- Comedy
- Horror
- Romance

We will compare classical machine learning models against TensorFlow sequence models, explain what drives their predictions, study where they disagree, and package the results in an interactive app that is easy to demonstrate in class.

## Why This Project Can Be Excellent

This dataset gives us a strong story:

- movie summaries are rich in narrative clues
- different genres use different lexical and emotional patterns
- classical models may be surprisingly strong
- neural models may capture context and sequence better
- poster images can provide a meaningful bonus multimodal comparison

That means we can deliver more than just accuracy numbers. We can deliver insight.

## What This Repository Includes

- exploratory data analysis
- text cleaning pipelines
- extractive and optional transformer summarization
- baseline models:
  - Multinomial Naive Bayes
  - Logistic Regression
  - Linear SVM
- TensorFlow LSTM text classifier
- evaluation reports and confusion matrices
- feature importance tables and word clouds
- disagreement analysis across models
- a Streamlit app scaffold for interactive prediction
- a bonus image-classifier starter for poster-based genre prediction

## Project Structure

```text
.
├── app/
│   └── streamlit_app.py
├── data/
├── models/
├── notebooks/
├── outputs/
│   ├── figures/
│   └── tables/
├── src/
│   ├── __init__.py
│   ├── app_helpers.py
│   ├── bonus_image_classifier.py
│   ├── config.py
│   ├── data_processing.py
│   ├── evaluate.py
│   ├── explain.py
│   ├── runtime.py
│   ├── summarize.py
│   ├── train_baselines.py
│   └── train_lstm.py
├── project_plan_option1.md
├── README.md
└── requirements.txt
```

## Expected Dataset

The pipeline expects a CSV or Parquet dataset with:

- one summary text column such as `summary`, `plot`, or `description`
- one genre label column such as `genre`

If the actual dataset uses different column names, pass them through the command line arguments.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Workflow

### 1. Preprocess and profile the dataset

```bash
python3 -m src.data_processing \
  --input data/movies.csv \
  --text-column summary \
  --label-column genre \
  --output-dir outputs
```

Outputs include:

- cleaned dataset preview
- dataset profile JSON
- summary statistics
- genre distribution figure
- summary length distribution figure

### 2. Train baseline models

```bash
python3 -m src.train_baselines \
  --input data/movies.csv \
  --text-column summary \
  --label-column genre \
  --output-dir outputs \
  --model-dir models
```

This trains:

- Multinomial Naive Bayes
- Logistic Regression
- Linear SVM

### 3. Train the TensorFlow LSTM

```bash
python3 -m src.train_lstm \
  --input data/movies.csv \
  --text-column summary \
  --label-column genre \
  --output-dir outputs \
  --model-dir models
```

This saves:

- the trained `.keras` model
- tokenizer
- label classes
- model summary
- epoch-by-epoch training plots

### 4. Generate interpretability and disagreement analysis

```bash
python3 -m src.explain \
  --input data/movies.csv \
  --text-column summary \
  --label-column genre \
  --model-dir models \
  --output-dir outputs
```

This creates:

- top feature tables
- word clouds
- disagreement table across models

### 5. Launch the interactive demo

```bash
streamlit run app/streamlit_app.py
```

### 6. Optional bonus: poster image classifier

```bash
python3 -m src.bonus_image_classifier \
  --input data/movies_with_images.csv \
  --image-column image_path \
  --label-column genre \
  --output-dir outputs \
  --model-dir models
```

## Assignment Checklist

This repository is designed to support every required item in the assignment:

- load and analyze the dataset
- clean the movie summaries
- build a summarization tool
- train at least four models
- evaluate overall and by genre
- plot TensorFlow training progress
- identify important words and create word clouds
- analyze disagreement cases
- build an interactive prediction tool
- include additional techniques from class
- leave room for the poster-image bonus

## Team Plan For Five People

To make this project organized and ambitious, here is the best five-person split. You can replace the placeholders below with actual names once the team agrees.

### Person 1: Data and EDA Lead

Focus:

- load the dataset
- inspect columns, missing values, duplicates, and class balance
- create summary-length statistics and genre-distribution visuals
- document data issues and cleaning decisions

How this person makes the project better:

- adds polished EDA figures for slides
- compares summary length by genre
- highlights dataset biases or imbalance early

### Person 2: Preprocessing and Summarization Lead

Focus:

- build the text cleaning pipeline
- compare light cleaning versus stronger classical cleaning
- implement the summarization tool
- create before-and-after examples for sample summaries

How this person makes the project better:

- tests whether summarized text still preserves genre signal
- compares extractive and transformer-style summaries if time allows

### Person 3: Classical Modeling Lead

Focus:

- train Naive Bayes, Logistic Regression, and Linear SVM
- compare TF-IDF settings, n-grams, and performance
- generate confusion matrices and evaluation tables

How this person makes the project better:

- tunes the strongest baseline carefully
- tests unigram vs. bigram features
- helps produce the most interpretable results

### Person 4: Deep Learning and Bonus Modeling Lead

Focus:

- train and tune the TensorFlow LSTM
- generate accuracy and loss plots over epochs
- analyze overfitting, dropout, and sequence length choices
- if time allows, build the poster-image classifier

How this person makes the project better:

- gives the project technical depth
- enables the optional multimodal comparison
- can test whether fusion improves performance

### Person 5: App, Presentation, and Integration Lead

Focus:

- build the Streamlit interface
- connect trained models to the demo
- organize outputs for the final presentation
- maintain the README, repo hygiene, and demo flow

How this person makes the project better:

- turns the project into something memorable and easy to present
- makes the final story coherent instead of fragmented
- ensures the repo is easy for teammates and graders to navigate

## How We Can Go Beyond the Standard

If we want this project to feel exceptional rather than merely complete, these are the highest-value upgrades:

- compare full summaries against generated short summaries
- add topic modeling by genre
- visualize text embeddings in 2D
- build a poster-image classifier
- combine text and image predictions for multimodal fusion
- create a taxonomy of model errors
- highlight movies where models strongly disagree
- compare human intuition to model predictions on a few examples

## Suggested Working Rhythm

To keep five people aligned without stepping on each other:

1. Lock the dataset schema and folder structure first.
2. Keep one shared issue list or task board.
3. Push small updates frequently instead of huge last-minute changes.
4. Save outputs in consistent folders so everyone can reuse them.
5. Meet once after EDA, once after baseline modeling, and once before recording the final demo.

## Recommended Final Deliverables

- cleaned and reproducible code in this repo
- saved model artifacts in `models/`
- visual outputs in `outputs/figures/`
- tables in `outputs/tables/`
- a polished Streamlit demo
- presentation slides
- recorded project walkthrough

## Notes

- `streamlit` is required for the app and may need to be installed in your local environment.
- transformer summarization is optional and only runs if the required libraries are installed.
- the image classifier is a bonus scaffold and expects a valid image-path column.

## Ambition Statement

The standard version of this project is a classifier. Our version should become a small, well-explained research story:

- what language signals genre
- which models perform best
- where models fail
- why they fail
- and how text and images together might improve prediction

That is the difference between finishing the assignment and delivering a project people remember.
