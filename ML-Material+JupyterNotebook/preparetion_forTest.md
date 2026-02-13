# Machine Learning Test Preparation – Quick Guide

## 1. What the Test Requires

You must be able to:

* **Write Python ML code independently**
* **Build, train, and evaluate models**
* **Fix incorrect base code**
* **Explain your decisions in Markdown**
* **Interpret model performance**

This is not memorization — it is **practical ML workflow understanding**.

---

# 2. Standard ML Workflow (VERY IMPORTANT)

Every task in the test follows the same structure:

## Step 1 — Load and Prepare Data

* Import libraries (`numpy`, `tensorflow`, `matplotlib`)
* Load dataset
* Normalize or preprocess data
* Split into **train / validation / test**

**Key idea:**
Good preprocessing → better learning.

---

## Step 2 — Create the Model

Choose architecture based on task:

* **Regression** → Dense output with linear activation
* **Binary classification** → `sigmoid` output
* **Multiclass classification** → `softmax` output
* **Images** → **CNN**

**Key idea:**
Model architecture must match the **data type**.

---

## Step 3 — Compile

You must correctly choose:

### Loss function

* Binary → `binary_crossentropy`
* Multiclass (integer labels) → `sparse_categorical_crossentropy`
* Regression → `mse`

### Optimizer

Usually:

* `Adam` (default good choice)

### Metrics

* `accuracy` for classification
* `mae` or `mse` for regression

---

## Step 4 — Train (Fit)

Important parameters:

* `epochs`
* `batch_size`
* `validation_split`

Watch for:

* **overfitting**
* **slow learning**
* **unstable loss**

---

## Step 5 — Evaluate

Always:

* Evaluate on **test set**
* Print **final accuracy/loss**
* Compare **train vs validation**

**Key idea:**
Test data must stay **unseen during training**.

---

# 3. CNN Concepts You MUST Know

## Why CNN instead of Dense for images

Dense layers:

* Flatten image → lose spatial structure
* Too many parameters
* Poor performance

CNN:

* Learn **edges → shapes → objects**
* Use **shared filters**
* Need **fewer parameters**
* Work much better on images

---

## CNN Core Layers

### Convolution

Detects visual patterns.

### ReLU

Adds non-linearity.

### Pooling

Reduces size and improves generalization.

### Dense + Softmax

Final classification.

---

# 4. Transfer Learning Essentials

## What it is

Using **pretrained ImageNet models** to solve new tasks.

## Two strategies

### Feature Extraction

* Freeze base model
* Train only new head
* Fast and stable

### Fine-tuning

* Unfreeze top layers
* Train with small learning rate
* Higher accuracy, risk of overfitting

---

## Why it works

Pretrained models already know:

* edges
* textures
* shapes

These appear in **almost all images**.

---

# 5. Overfitting & How to Fix It

Signs:

* Train accuracy ↑
* Validation accuracy ↓

Fixes:

* More data
* Data augmentation
* Dropout
* EarlyStopping
* Smaller model

---

# 6. What You MUST Write in Markdown During Test

## Explain:

### Design choices

Why this model?
Why this loss?
Why this preprocessing?

### Results

* Train accuracy
* Validation accuracy
* Test accuracy

### Improvements tried

* Changed epochs
* Added dropout
* Used CNN / transfer learning

### Bug fixing

Explain **what was wrong in base code** and **how you fixed it**.

---

# 7. Clean Notebook Structure (Important for Score)

## Recommended layout

### 1. Title + Task description

### 2. Imports

### 3. Data loading & preprocessing

### 4. Model creation

### 5. Training

### 6. Evaluation

### 7. Analysis in Markdown

Keep everything:

* **clear**
* **short**
* **well formatted**

---

# 8. Golden Rules for the Test

* Always normalize image data.
* Match **output layer + loss function**.
* Never evaluate on training data.
* Use validation to detect overfitting.
* Write explanations in Markdown.
* Keep code clean and readable.

---

# Final Thought

The test checks **real ML thinking**, not memorization.

If you can:

* build a model
* train it correctly
* evaluate honestly
* explain your reasoning

→ **you will pass.**
