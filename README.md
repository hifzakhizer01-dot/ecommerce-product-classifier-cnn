# 🛍️ E-Commerce Product Classifier using CNN

A Deep Learning project that classifies e-commerce product images into 4 categories using a Convolutional Neural Network (CNN) trained on 44,288 real Myntra products.

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| Test Accuracy | 93.85% |
| Total Images | 44,288 |
| Categories | 4 |
| Best Epoch | 7 |

---

## 🏷️ Categories
- 👜 Accessories
- 👕 Apparel
- 👟 Footwear
- 🧴 Personal Care

---

## 🧠 CNN Architecture

| Layer | Details |
|-------|---------|
| Conv2D Block 1 | 32 filters, 3x3, ReLU |
| Conv2D Block 2 | 64 filters, 3x3, ReLU |
| Conv2D Block 3 | 128 filters, 3x3, ReLU |
| Batch Normalization | After each Conv block |
| MaxPooling | 2x2 after each block |
| Dropout | 0.5 |
| Dense | 256 neurons, ReLU |
| Output | 4 neurons, Softmax |

---

## 🔧 Techniques Used
- Convolutional Neural Network (CNN)
- Batch Normalization
- Data Augmentation (flip, rotation, zoom)
- Dropout Regularization
- Early Stopping
- Adam Optimizer

---

## 📁 Dataset
- **Source:** Myntra Fashion Products (Kaggle)
- **Link:** [Fashion Product Images Small](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small)
- **Size:** 44,288 product images
- **Image Size:** 64x64 pixels

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/hifzakhizer01-dot
/ecommerce-product-classifier-cnn.git
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit app
```bash
streamlit run app.py
```

---

## 📈 Results

### Classification Report
| Category | Precision | Recall | F1-Score |
|----------|-----------|--------|----------|
| Accessories | 83% | 96% | 89% |
| Apparel | 98% | 95% | 97% |
| Footwear | 99% | 95% | 97% |
| Personal Care | 98% | 65% | 78% |

---

## 💼 Marketing Relevance
This project demonstrates how AI can automate product categorization for e-commerce platforms — the same technology used by Amazon and Daraz to organize millions of products automatically, saving hours of manual work for marketing teams.

---

## 🛠️ Tech Stack
- Python
- TensorFlow / Keras
- Streamlit
- OpenCV
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

---

## 👤 Author
**[Hifza Khizer]**
Business Data Analytics Student 

---

## 📚 References
- [Kaggle Dataset](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Streamlit Documentation](https://streamlit.io/)
