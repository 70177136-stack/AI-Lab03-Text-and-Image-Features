# Lab 03: Transforming Textual and Image Data into Machine-Understandable Formats

**Author:** Mohsin Raza  
**SAP ID:** 70177136  
**Email:** 70177136@student.uol.edu.pk  
**Institution:** The University of Lahore (UOL)  
**Course:** Introduction to Artificial Intelligence (AI-101L)  

---

## 📌 Overview
This repository contains the full implementation and analysis for **Laboratory 03**:
1. **Text Transformation:** Converting text strings to Bag-of-Words vectors using `CountVectorizer`.
2. **N-Gram Expansion:** Evaluating feature vocabulary growth from unigram `(1, 1)` to bigram `(1, 2)`.
3. **Image Transformation:** Normalizing image channels, spatial resizing ($64 \times 64$), grayscale conversion ($3 \to 1$), and 1-D vector flattening ($4096$ dimensions).
4. **Dataset Serialization:** Assembling a 20-image 2-class dataset into binary NumPy format (`dataset.npz`).
5. **Dimensionality Analysis:** In-depth discussion of the Curse of Dimensionality ($N = 20, D = 4096$).

---

## 📁 Repository Structure
```
Lab03_Text_and_Image_Features/
├── Lab_03_Text_and_Image_Features.ipynb   # Main notebook with all answered questions
├── image_utils.py                         # Reusable image-to-vector transformation function
├── dataset.npz                            # 20-image dataset (X: 20x4096, y: 20)
├── data/                                  # Source images organized by class
│   ├── class_a/                           # Class 0: Circular geometric patterns
│   └── class_b/                           # Class 1: Cross-box geometric patterns
├── report.md                              # Detailed formal lab report
├── requirements.txt                       # Python dependencies
└── README.md                              # Repository guide
```

---

## 🚀 Execution Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Verify Image Processing Module
```bash
python image_utils.py
```

### 3. Run the Jupyter Notebook
```bash
jupyter notebook Lab_03_Text_and_Image_Features.ipynb
```
