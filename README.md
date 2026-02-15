# Sport vs Politics News Classification

## 1. Problem Statement
The goal is to classify news articles as **Sport** or **Politics** using classical machine learning techniques implemented from scratch (Python standard library only).

## 2. Dataset
- BBC News dataset (2225 articles, 5 categories: business, entertainment, politics, sport, tech). [web:40][web:42]
- For this project, only the `sport` and `politics` categories are used.
- Original data link: http://mlg.ucd.ie/files/datasets/bbc-fulltext.zip

Expected folder structure:

data/
bbc/
sport/
<sport article .txt files>
politics/
<politics article .txt files>

text

## 3. Methods

### Preprocessing
- Lowercasing
- Regex tokenization using `\w+`
- No external NLP libraries

### Features
- Bag of Words (BoW) with a vocabulary built on training data
- TF-IDF (Term Frequency-Inverse Document Frequency)

### Models
All models implemented from scratch using only Python standard library:

- **Multinomial Naive Bayes** with BoW
- **Perceptron (linear classifier)** with BoW
- **Perceptron (linear classifier)** with TF-IDF

## 4. How to Run

1. Download and unzip the BBC dataset.
2. Place `sport` and `politics` folders under `data/bbc/`.

Example:

data/
bbc/
sport/
politics/

text

3. Run the classifier script:

```bash
python src/B23CM1040_prob4.py
The script:

Loads the data

Builds the vocabulary and features

Trains all three models

Prints accuracies for:

Naive Bayes + BoW

Perceptron + BoW

Perceptron + TF-IDF

5. Results
Model	Features	Accuracy
Multinomial Naive Bayes	BoW	YOUR_VALUE
Perceptron	BoW	YOUR_VALUE
Perceptron	TF-IDF	YOUR_VALUE
6. Limitations and Future Work
Bag-of-words and TF-IDF ignore word order and deeper semantics.

Only BBC news; models may not generalize to social media or other domains. [web:42]

No hyperparameter tuning or cross-validation.

Future work: add n-grams, regularization, or more advanced models if libraries are allowed.

text

Just fill in your actual accuracies and adjust the file names (roll number, etc.).

### 3. Optional: GitHub Pages

If your teacher really wants a “GitHub Page” (a web view), you can:

1. Go to your repo → Settings → “Pages”.
2. Choose source: `Deploy from branch` → `main` → `/root` (or `/docs`, if you create a `docs/` folder).
3. Save. GitHub will give you a URL like `https://username.github.io/sport-politics-classification/`.

You don’t need anything fancy: just make sure `README.md` (or `docs/index.md`) is clear, with:

- Short description of problem
- Dataset description
- Methods (features + models)
- Results table
- Link to the full report PDF

That’s enough to satisfy “GitHub page with all details” for this kind of assignment.