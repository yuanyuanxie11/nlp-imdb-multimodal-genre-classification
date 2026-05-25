# Text Analytics Final Project Plan

## Assignment Interpreted

This plan is built for **Option 1: Visual and Genre Classification** using the IMDB multimodal dataset with four genres:

- Action
- Comedy
- Horror
- Romance

The assignment requires us to analyze the data, clean the movie summaries, build a summarization tool, train at least four genre-classification models, evaluate them carefully, explain disagreements between models, and deliver an interactive demo plus a presentation and video.

## Project Goal

Build a multimodal movie-genre analysis project that:

- predicts genre from plot summaries
- compares traditional NLP models with deep learning
- explains which words drive predictions
- demonstrates model behavior on real and fake summaries
- optionally extends into image-based poster classification

## Recommended Project Framing

### Main question

How accurately can we predict whether a movie belongs to Action, Comedy, Horror, or Romance using only its text summary, and what linguistic patterns distinguish the genres?

### Stronger framing for presentation

We are not just building a classifier. We are comparing **interpretable classical NLP methods** against **neural sequence models**, then using those differences to learn how genres are expressed in movie summaries.

## Exact Plan For Every Assignment Requirement

### 1. Load in the data and analyze each column

Plan:

- Load the dataset into a Pandas DataFrame.
- Inspect shape, missingness, duplicate rows, class balance, and column types.
- For the text summary column, compute:
  - average word count
  - median word count
  - shortest and longest summaries
  - vocabulary size
  - most common words before cleaning
- For image fields, verify file paths or URLs and count valid image records.

Deliverables:

- table of columns and meanings
- missing-value summary
- genre distribution bar chart
- summary-length histogram
- sample records from each genre

Bonus idea:

- Compare summary length by genre. Horror and romance may show different average descriptive styles.

### 2. Clean the movie summary column

Plan:

- lowercase text
- remove HTML, punctuation, extra whitespace, and strange symbols
- expand contractions if useful
- optionally remove stopwords for classical models only
- optionally lemmatize text
- preserve a second version with minimal cleaning for neural models

Important modeling choice:

- Use **two cleaning pipelines**:
  - classical pipeline for Naive Bayes and Logistic Regression
  - light-cleaning pipeline for LSTM so word order and syntax remain more natural

Deliverables:

- written explanation of cleaning choices
- before/after examples for 3 to 5 summaries

Bonus idea:

- Compare performance of stopword removal vs. no stopword removal. That can become a nice small experiment.

### 3. Build a summarization tool and summarize a few movies

Plan:

- Build an extractive or abstractive summarization component.
- Easiest practical route:
  - use a pretrained summarization model from Hugging Face for demo summaries
  - or use TextRank / frequency-based extractive summarization as a lightweight baseline
- Show original summary + compressed summary for several examples from different genres.

Recommended approach:

- Implement two summarizers if time allows:
  - baseline extractive summarizer
  - pretrained transformer summarizer
- Briefly compare which is more readable or more genre-preserving.

Deliverables:

- notebook section with sample summaries
- interactive tool button: "Summarize this movie plot"

Bonus idea:

- Test whether classification accuracy changes if the model uses the original summary versus the generated short summary.

### 4. Build at least four models

Required model mix from the assignment:

- at least one Naive Bayes model
- at least one generalized linear model such as Logistic Regression
- at least one TensorFlow model such as an LSTM
- total of at least four models

Recommended model set:

1. Multinomial Naive Bayes with TF-IDF features
2. Logistic Regression with TF-IDF features
3. Linear SVM with TF-IDF features
4. TensorFlow LSTM with tokenized sequences and embeddings

Optional fifth model if time allows:

- TensorFlow CNN for text classification
- or Random Forest on reduced TF-IDF features

Why this set is strong:

- Naive Bayes satisfies the requirement and is a clean baseline
- Logistic Regression is interpretable and usually strong for text classification
- SVM often performs very well on sparse text data
- LSTM satisfies the deep learning requirement and adds sequential modeling

Bonus idea:

- Add a late-fusion multimodal model combining text prediction with poster-image prediction

### 5. Evaluate the models in terms of overall accuracy and accuracy by genre

Plan:

- Use train/validation/test splits or stratified cross-validation
- Report:
  - overall accuracy
  - macro F1
  - weighted F1
  - per-genre precision, recall, F1
  - confusion matrix
- Compare where models struggle:
  - Action vs Horror
  - Romance vs Comedy
  - or other confusing pairs found in the data

Deliverables:

- model comparison table
- confusion matrices for all models
- bar chart of per-genre accuracy

Bonus idea:

- Include calibration or confidence analysis. Show that some wrong predictions were made with low confidence.

### 6. For TensorFlow models, plot the models and plot accuracy over epochs

Plan:

- Use `model.summary()`
- use `tf.keras.utils.plot_model()` if available
- track training and validation accuracy and loss over epochs
- discuss overfitting and early stopping

Deliverables:

- architecture diagram
- training-vs-validation accuracy plot
- training-vs-validation loss plot

Bonus idea:

- Add dropout and compare overfitting before vs. after regularization

### 7. Show the most important words from each model and build a word cloud

Plan:

- For Naive Bayes:
  - inspect top log-probability words for each genre
- For Logistic Regression or Linear SVM:
  - inspect highest positive coefficients per class
- For LSTM:
  - use approximation methods such as:
    - attention layer if you include one
    - gradient-based saliency
    - LIME or SHAP for local explanations

Deliverables:

- top-10 or top-20 words per genre for each interpretable model
- one word cloud per genre
- one short explanation of why these words make sense

Important note:

- Word clouds look good in slides, but coefficient tables are more rigorous. Use both.

Bonus idea:

- Compare "genre stereotypes" from the model to actual film-language patterns. That adds a more analytical discussion.

### 8. Find movies that had different predictions across models

Plan:

- Create a disagreement table:
  - true genre
  - summary text
  - Naive Bayes prediction
  - Logistic Regression prediction
  - SVM prediction
  - LSTM prediction
- Select 5 to 10 interesting examples
- For each case, explain which words or phrases may have pushed models in different directions

Best kinds of examples:

- genre-blended plots
- summaries with ambiguous emotional tone
- summaries using action vocabulary in romance contexts
- comedy plots with dark or horror-like language

Deliverables:

- error-analysis section
- one slide on "Why the models disagree"

Bonus idea:

- Cluster the disagreements into categories like ambiguity, sparse cues, misleading keywords, or short-summary bias

### 9. Build an interactive tool

Plan:

- Use **Streamlit** for speed and clean presentation
- User inputs:
  - movie summary text
  - optional toggle for cleaning
  - model selector
- Outputs:
  - predicted genre
  - class probabilities
  - cleaned text preview
  - optional generated summary
  - optional explanation words highlighted

Recommended tabs:

- `Dataset Explorer`
- `Summarizer`
- `Genre Predictor`
- `Model Comparison`

Deliverables:

- runnable Streamlit app
- short README instructions

Bonus idea:

- Add a "Fake Movie Generator" mode where users type a wild plot and compare model behavior

### 10. Add other techniques taught in the course

Strong candidates:

- n-grams
- TF-IDF weighting
- topic modeling
- sentiment analysis
- named entity recognition
- dimensionality reduction with PCA or t-SNE on text embeddings
- SHAP or LIME interpretability
- class-imbalance handling

Best recommendation:

- Add **topic modeling** by genre and **embedding visualization** of summaries in 2D. These are visually compelling and academically relevant.

Bonus idea:

- Compare unigram vs. bigram models to show how phrases like "serial killer" or "falls in love" matter more than single words.

### 11. Present findings in a 10-minute video with slides, sample code, and interactive demo

Recommended slide flow:

1. Problem and dataset
2. Why genre classification is interesting
3. Data overview and class balance
4. Text cleaning and preprocessing
5. Summarization tool
6. Model lineup
7. Results comparison
8. Important words and interpretability
9. Error analysis and model disagreements
10. Interactive demo
11. Bonus work and future improvements
12. Final takeaways

Recommended time split:

- 1 min intro and dataset
- 2 min preprocessing and summarization
- 3 min models and metrics
- 2 min interpretability and disagreement analysis
- 2 min live demo and conclusion

## Best End-To-End Workflow

### Phase 1. Data setup

- download dataset
- inspect schema
- verify text and image availability
- split train/validation/test with stratification

### Phase 2. EDA and preprocessing

- analyze class balance and text length
- create cleaning pipelines
- save cleaned data artifacts

### Phase 3. Summarization

- build baseline summarizer
- optionally add transformer summarizer
- choose example summaries

### Phase 4. Modeling

- fit TF-IDF vectorizer
- train Naive Bayes
- train Logistic Regression
- train SVM
- train LSTM

### Phase 5. Evaluation and interpretation

- generate metrics and confusion matrices
- extract important words
- build word clouds
- analyze disagreement cases

### Phase 6. Interactive app

- connect trained models
- add user input and prediction display
- add summarization tab

### Phase 7. Presentation packaging

- finalize charts
- record demo
- clean repo and write README

## Recommended Repo Structure

```text
final-project/
  data/
  notebooks/
  src/
    data_processing.py
    summarize.py
    train_baselines.py
    train_lstm.py
    evaluate.py
    explain.py
    app_helpers.py
  models/
  outputs/
    figures/
    tables/
  app/
    streamlit_app.py
  README.md
  requirements.txt
```

## Suggested Team Division

If you have 4 to 5 people:

- Person 1: data loading, EDA, cleaning
- Person 2: summarization and baseline models
- Person 3: Logistic Regression and SVM evaluation
- Person 4: TensorFlow model and plots
- Person 5: Streamlit app, slides, and final integration

If only 4 people, combine Person 2 and Person 3.

## What A Strong Final Result Looks Like

At minimum, a strong project should include:

- clean and reproducible preprocessing
- four models with clear comparison
- one good TensorFlow model with training curves
- thoughtful per-genre evaluation
- meaningful explanation of feature importance
- polished Streamlit demo
- concise presentation with a few memorable visuals

## Highest-Value Bonus Ideas

These are the best bonus ideas if you want to stand out.

### Bonus 1. Poster-image classifier

- Train a CNN on poster images alone
- Compare image-only accuracy against text-only accuracy
- Ask whether posters or summaries are more informative for genre

### Bonus 2. Multimodal fusion

- Combine poster-image predictions with text predictions
- Simple fusion method:
  - average probabilities from text and image models
- Strong story:
  - "Text and visuals capture different genre signals"

### Bonus 3. Genre ambiguity score

- Create a measure of how strongly the models disagree
- Highlight movies with blended genre signals

### Bonus 4. Human-vs-model test

- Ask classmates to classify a few summaries
- Compare human confusion to model confusion

### Bonus 5. Summary compression experiment

- Classify full summaries versus compressed summaries
- This makes the summarization component more integrated into the project

### Bonus 6. Embedding visualization

- Use sentence embeddings and plot summaries in 2D
- Show whether genres naturally cluster

### Bonus 7. Error taxonomy

- Categorize failure cases
- This adds maturity beyond reporting accuracy alone

## Practical Recommendations

- Use **TF-IDF + Logistic Regression** as the likely strongest classical baseline.
- Use **LSTM with pretrained embeddings or trainable embeddings** as the required deep model.
- Use **Streamlit** rather than Flask unless someone on the team already prefers Flask.
- Save every trained model and preprocessing object with consistent naming.
- Lock your train/test split early so comparisons stay fair.

## Likely Presentation Takeaways

These are the kinds of conclusions you will probably be able to support:

- Classical linear models may perform surprisingly well on genre text classification.
- Some genres are easier to classify because their summaries contain stronger lexical signals.
- Deep learning may help with sequence context, but not always beat TF-IDF baselines on modest datasets.
- Model disagreement is informative and often reveals genre overlap rather than pure model failure.
- Poster images and plot summaries may capture complementary information.

## Recommended "Perfect Answer" Strategy

To fully satisfy the assignment and impress the instructor:

- explicitly map each notebook requirement to one output in your repo or presentation
- include both performance metrics and interpretation
- include at least one thoughtful experiment beyond the minimum
- make the app polished and easy to demo
- tell a clear story, not just a list of models

## Short Version

If you want the safest high-scoring version of this project, do this:

1. Clean summaries with two pipelines
2. Build TF-IDF Naive Bayes, Logistic Regression, and SVM
3. Build a TensorFlow LSTM
4. Evaluate with accuracy, F1, and confusion matrices
5. Extract important words and create word clouds
6. Analyze disagreement cases
7. Build a Streamlit app with summarization and prediction
8. Add one bonus experiment, ideally poster classification or multimodal fusion

