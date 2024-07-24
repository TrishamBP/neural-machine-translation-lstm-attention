# English to Portuguese Neural Machine Translation

This project implements a neural machine translation system that translates English sentences to Portuguese using a sequence-to-sequence model with attention mechanism.

## Table of Contents
- [Overview](#overview)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Files Description](#files-description)
- [Training](#training)
- [Inference](#inference)
- [Minimum Bayes Risk Decoding](#minimum-bayes-risk-decoding)
- [Results](#results)


## Overview

This neural machine translation system is built using TensorFlow and implements a sequence-to-sequence model with attention for translating English sentences to Portuguese. The model uses an encoder-decoder architecture with cross-attention and implements various techniques such as Minimum Bayes Risk (MBR) decoding for improved translation quality.

## Model Architecture

The model consists of the following main components:

1. **Encoder**: Bidirectional LSTM that processes the input English sentence.
2. **Decoder**: LSTM-based decoder with cross-attention mechanism.
3. **Cross-Attention**: Multi-head attention layer for attending to relevant parts of the encoded input.

![image](https://github.com/user-attachments/assets/54b808ef-9263-439e-a963-9a1be4557f3e)

## Installation

To set up the project, follow these steps:

1. Clone the repository:
```agsl
git clone https://github.com/TrishamBP/neural-machine-translation-lstm-attention.git
```
2. Install the required dependencies:
```pip install -r requirements.txt```
## Usage

To translate an English sentence to Portuguese:

1. Ensure you have a trained model saved as 'translator_model'.
2. Run the main script:

```python main.py```

4. The script will output multiple translation candidates and the selected best translation.

## Files Description

- `main.py`: The main script for running translations and MBR decoding.
- `utils.py`: Utility functions for data loading, preprocessing, and evaluation metrics.
- `training.py`: Script for training the translator model.
- `translator.py`: Defines the main Translator model.
- `encoder.py`: Implementation of the Encoder class.
- `decoder.py`: Implementation of the Decoder class.
- `cross_attention.py`: Implementation of the CrossAttention layer.

## Training

The model is trained using the following process:

1. Data is loaded and preprocessed from a Portuguese-English parallel corpus.
2. The model is compiled with Adam optimizer and custom loss and accuracy functions.
3. Training is performed with early stopping based on validation loss.

To train the model:

```
python training.py
```

## Inference

The trained model can be used for inference as follows:

1. Load the trained model.
2. Use the `translate` function to generate a translation for an input English sentence.
3. Optionally use MBR decoding for improved translation quality.

## Minimum Bayes Risk Decoding

This project implements Minimum Bayes Risk (MBR) decoding to improve translation quality:

1. Multiple translation candidates are generated.
2. Candidates are scored based on their similarity to other candidates.
3. The candidate with the highest average similarity is selected as the final translation.

## Results
![image](https://github.com/user-attachments/assets/e204262f-dbb0-4f73-92ab-b30c0c6505d4)
