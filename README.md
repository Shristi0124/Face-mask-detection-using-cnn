# Face-mask-detection-using-cnn

---

````markdown
# 😷 Face Mask Detection using CNN

This project implements a deep learning-based face mask detection system using Convolutional Neural Networks (CNN). It classifies whether a person is **wearing a mask** or **not wearing a mask** in real-time using a webcam or image inputs.

---

## 📌 Features

- Binary classification: Mask vs No Mask
- Real-time face mask detection via webcam
- Built with TensorFlow / Keras and OpenCV
- Trained on a balanced dataset with masked and unmasked faces

---

## 📁 Dataset

We used the [Face Mask Dataset](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset) from Kaggle, which contains:

- `With Mask`: 7,000+ images
- `Without Mask`: 7,000+ images

**Preprocessing Steps:**

- Resized all images to `150x150`
- Normalized pixel values
- Data augmentation (rotation, zoom, flip) for better generalization

---

## 🧠 Model Architecture

The CNN model consists of:

- 3 Convolutional + MaxPooling layers
- Flatten layer
- Dense layer with ReLU activation
- Output layer with Sigmoid activation

```python
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2,2),
    
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
````

---

## ⚙️ Installation & Usage

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/face-mask-detection-using-cnn.git
cd face-mask-detection-using-cnn
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the model

```bash
python train_model.py
```

### 4. Test on image

```bash
python predict_image.py --image test.jpg
```

### 5. Real-time detection with webcam

```bash
python detect_mask_webcam.py
```

---

## 📈 Results

* Accuracy: \~95% on test data
* Loss and accuracy plots included in the results folder

---

## 🛠️ Tools & Technologies

* Python
* TensorFlow / Keras
* OpenCV
* NumPy
* Matplotlib

---

## 📌 Future Improvements

* Improve accuracy with larger datasets
* Deploy on edge devices (e.g., Raspberry Pi)
* Integrate with alert systems or IoT

---

## 🙋‍♂️ Author

**Shristi**
📍 Dehradun, Uttarakhand
📧 [shristishristi59@gmail.com](mailto:shristishristi59@gmail.com)
🔗 [LinkedIn](https://github.com/Shristi0124) | [GitHub](https://github.com/Shristi0124)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
