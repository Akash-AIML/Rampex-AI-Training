
# Naive Bayes – Spam Email Classification (Day 3 Progress)

This folder records my Day 3 progress in machine learning training.  
Today I focused on building spam–ham classifiers using **Naive Bayes**, a classic probabilistic model widely used for text classification.

---

## 🔹 What I Worked On Today

- Loaded the same spam/ham email dataset  
- Preprocessed text and removed unused columns  
- Converted email text into numerical features using **TF-IDF**  
- Trained two Naive Bayes models:
  - **MultinomialNB**
  - **BernoulliNB**
- Compared their accuracy, precision, recall, and F1-scores  
- Observed which version performs better for this dataset  

---

## 🔹 Results

- **MultinomialNB** performed strongly on TF-IDF features  
- **BernoulliNB** also worked well but slightly less accurate  
- Both models handled text classification efficiently  
- Predictions were fast, and training time was extremely low  
- Achieved performance in the **96–98% range**, depending on parameters  

---

## 🔹 Files in This Folder



MultinomialNB.ipynb # Notebook using Multinomial Naive Bayes
BernoulliNB.ipynb # Notebook using Bernoulli Naive Bayes
spam_ham_dataset.csv # Dataset used for both models
README.md # Day 3 progress log


---

## 🔹 Next Steps

- Try smoothing parameters (e.g., different alpha values)  
- Compare Naive Bayes results with Logistic Regression from Day 2  
- Add visualization and confusion matrices  
- Begin preparing for SVM or Decision Tree models in upcoming days  

---

This README documents my Day 3 progress in implementing and comparing Naive Bayes classifiers.
