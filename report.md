# Laboratory Report: Lab 03 — Transforming Textual and Image Data into Machine-Understandable Formats

**Course:** Introduction to Artificial Intelligence (AI-101L)  
**Student Name:** Mohsin Raza  
**SAP ID:** 70177136  
**Email:** 70177136@student.uol.edu.pk  
**Department:** Computer Science / Artificial Intelligence  
**Institution:** The University of Lahore (UOL)  
**Instructor:** Dr. Muhammad Awais  

---

## 1. Introduction and Objectives

Machine learning models are strictly mathematical estimators; they cannot natively consume raw ASCII text strings or multi-megabyte JPEG camera files. Both textual documents and visual images must undergo deterministic feature engineering transformations that map unstructured modalities into fixed-dimensional numeric matrices ($\mathbb{R}^{N \times D}$).

This laboratory investigated:
1. Converting text documents into sparse coordinate representations via Bag-of-Words (`CountVectorizer`).
2. Quantifying vocabulary growth across unigram vs unigram+bigram configurations.
3. Transforming images through grayscale conversion ($3 \to 1$ channels), spatial resizing ($64 \times 64$), and 1-D vector flattening ($4096$ dimensions).
4. Assembling and persisting a 20-sample multi-class visual dataset (`dataset.npz`).
5. Critical statistical evaluation of the **Curse of Dimensionality** ($N = 20, D = 4096$).

---

## 2. Answers to Discussion Questions

### Part A: Textual Transformations (Bag-of-Words)

#### Question 1: What does each column in the resulting matrix represent?
Each column corresponds to a distinct vocabulary token identified across the corpus. The value at cell $(i, j)$ records the absolute frequency with which token $j$ appears in document $i$.

#### Question 2: In this specific numerical matrix, what is the machine-understandable format?
A 2-D matrix of dimensions $(N \times V)$ where each row is a fixed-length numerical feature vector in $\mathbb{R}^V$. This algebraic representation enables vector space comparisons (e.g. cosine similarity, Euclidean distance, and dot products with weight vectors).

#### Question 3: How is this format an oversimplification of original text data?
1. **Grammar & Word Order Obliteration:** Phrases with opposite meanings like *"good, not bad"* and *"bad, not good"* produce identical Bag-of-Words representations.
2. **Semantic Independence Assumption:** Orthogonal vector axes treat synonyms (*"automobile"* and *"vehicle"*) as completely unrelated dimensions.
3. **Document Length Sensitivity:** Longer documents accumulate larger raw counts, artificially skewing dot product magnitudes without conveying greater semantic density.

---

### Part B: Visual Transformations (Image to Feature Vector)

#### Question 1: Explain how grayscale conversion acts as a feature engineering technique.
Grayscale conversion computes perceptual luminance:
$$Y = 0.299R + 0.587G + 0.114B$$
By collapsing three color channels (RGB) into one intensity value per pixel, it reduces the input dimensionality by $66.7\%$ while preserving edge contours, topological forms, and textural gradients essential for geometric pattern recognition.

#### Question 2: If the loaded image had dimensions $H \times W$, what is the length of the final flattened vector?
$$\text{Vector Length} = H \times W$$

#### Question 3: Why is flattening necessary before feeding data to simple models (Logistic Regression, SVM)?
Linear algorithms compute inner products between an input vector $\mathbf{x} \in \mathbb{R}^D$ and parameter weights $\mathbf{w} \in \mathbb{R}^D$:
$$\hat{y} = \sigma(\mathbf{w}^T \mathbf{x} + b)$$
They do not support 2-D spatial matrix convolutions. Flattening unwraps 2-D image pixels into an indexed 1-D vector matching the model's weight dimensions.

#### Question 4: What is the advantage of resizing all images to a consistent size (e.g. $64 \times 64$)?
- **Architectural Determinism:** Classifiers require an invariant input dimension ($D = 4096$). Unstandardized image resolutions would cause matrix shape mismatch exceptions.
- **Batch Processing:** Consistent sizing allows uniform memory batching on modern CPU/GPU caches.

---

## 3. Analysis: 20 Samples in 4096 Dimensions (Curse of Dimensionality)

Building a machine learning model on $N = 20$ samples across $P = 4096$ dimensions represents a severe violation of statistical learning bounds ($P \gg N$). 

In this regime:
1. **High Sample Sparsity:** A 4096-dimensional space is virtually empty; data points are isolated outliers at the extreme periphery.
2. **Distance Concentration:** Euclidean distance between any pair of points converges to nearly the same value, rendering distance-based classifiers ($k$-NN, kernel SVM) ineffective.
3. **Catastrophic Overfitting:** A linear classifier has 4096 adjustable weights with only 20 constraints. The model can easily memorize spurious training noise with 100% training accuracy but will completely fail to generalize on test instances.

**Solutions:**
- Perform dimensionality reduction (e.g. PCA to reduce 4096 dimensions to 5–10 principal components).
- Utilize Convolutional Neural Networks (CNNs) which share small kernels across space rather than assigning independent parameters to each raw pixel.

---

## 4. Deliverables Checklist
- `Lab_03_Text_and_Image_Features.ipynb`: Complete notebook with all questions and exercises.
- `image_utils.py`: Reusable, documented `image_to_features` pipeline.
- `data/class_a/` & `data/class_b/`: 20 standardized PNG images across 2 distinct visual classes.
- `dataset.npz`: Compact NumPy binary package containing $X$ $(20, 4096)$ and $y$ $(20,)$.
- `report.md`: Formal lab report.
