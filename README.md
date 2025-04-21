# Grammar Scoring System

## Project Overview

This project develops a system to automatically score the grammatical quality of speech transcriptions. The system uses a deep learning approach combining BERT embeddings with convolutional neural networks (CNN) to predict grammar quality scores.

## Workflow

1. **Audio Processing**: Audio files (.wav) are transcribed using the OpenAI Whisper model.
2. **Data Cleaning**: Noisy audio samples were identified and removed from the dataset.
3. **Text Embedding**: BERT embeddings are used to convert text transcriptions into meaningful vector representations.
4. **Model Development**: CNN and LSTM architectures were developed and compared for predicting grammar scores.
5. **Inference**: The trained model is used to score new, unlabeled transcriptions.

## Noise Handling Approach

Initially, I attempted to classify noisy samples using LLaMA 3.3 via the Groq API. However, this approach resulted in numerous misclassifications. Since the dataset was relatively small (444 samples), I opted for manual classification of noisy samples.

I explored cleaning noisy samples by adjusting playback speed and pitch, but this proved impractical as each audio file required different adjustment parameters. Ultimately, I decided to remove noisy samples entirely from the training dataset to ensure model quality.

## Model Architecture

The system implements two model architectures:

1. **BERT + LSTM**: Uses bidirectional LSTM layers on top of BERT embeddings.
2. **BERT + CNN**: Uses multiple convolutional layers with varying kernel sizes to capture different n-gram features from BERT embeddings.

Comparative evaluation showed that the CNN model performed better for this specific task.

## Requirements

- Python 3.8+
- PyTorch
- Transformers (Hugging Face)
- Pandas
- NumPy
- TensorBoard
- tqdm
- scikit-learn
- OpenAI Whisper (for audio transcription)

## TensorBoard Visualization

Training progress and model comparisons can be visualized using TensorBoard:

```bash
tensorboard --logdir=runs
```
