# Grammar Scoring System

## Project Overview

This project develops a system to automatically score the grammatical quality of speech transcriptions. The system uses a deep learning approach that combines BERT embeddings with convolutional neural networks (CNN) to predict grammar quality scores.

> 📓 **Note**: For implementation details and code, please refer to the accompanying `.ipynb` notebook. The model was trained using a Tesla P100 GPU provided by Kaggle.

---

## Workflow

1. **Audio Processing**  
   Audio files (`.wav`) are transcribed using the OpenAI Whisper model.

2. **Data Cleaning**  
   Noisy audio samples were identified and removed from the dataset.

3. **Text Embedding**  
   BERT embeddings are used to convert text transcriptions into meaningful vector representations.

4. **Model Development**  
   CNN and LSTM architectures were developed and compared for predicting grammar scores.

5. **Inference**  
   The trained model is used to score new, unlabeled transcriptions.

---

## Noise Handling Approach

Initially, I attempted to classify noisy samples using **LLaMA 3.3 via the Groq API**. However, this method resulted in numerous misclassifications. Given the relatively small dataset size (444 samples), I opted for **manual classification** of noisy samples.

I explored cleaning noisy samples by adjusting **playback speed and pitch**, but this proved impractical as each audio file required different adjustment parameters. Ultimately, I **removed noisy samples** entirely from the training dataset to ensure model quality.

---

## Model Architecture

The system implements and compares two deep learning architectures:

1. **BERT + LSTM**  
   Applies bidirectional LSTM layers on top of BERT embeddings to capture sequential dependencies.

2. **BERT + CNN**  
   Applies multiple convolutional layers with varying kernel sizes to extract n-gram features from BERT embeddings.

📈 **Evaluation Result**: The **BERT + CNN** model outperformed the LSTM-based model in this specific task.

---

## Requirements

- Python 3.8+
- PyTorch
- Hugging Face Transformers
- Pandas
- NumPy
- TensorBoard
- tqdm
- scikit-learn
- OpenAI Whisper

---

## TensorBoard Visualization

Training progress and model comparisons can be visualized using TensorBoard:

```bash
tensorboard --logdir=runs
