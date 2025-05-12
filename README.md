# Pokémon Card PSA Grader

Contributors: Aditya Matiwala, Adrian Lara, Kashvi Panjolia

A system for automatically estimating PSA grades of Pokémon cards from images using a ResNet18 Model

## Overview

This project uses a ResNet18-based convolutional neural network to predict the PSA grade (1-10) of Pokémon cards from images. The system:

1. Preprocesses card images to align and normalize them
2. Extracts visual features relevant to grading (corners, edges, centering, etc.)
3. Feeds these into a trained neural network to predict the PSA grade

## Installation

### Requirements

To install all dependencies:

```bash
pip install -r requirements.txt
```

## Model Training

To train the model:

```bash
python train_model.py
```

## Using the Grader

To predict the PSA grade of a single card image:

```bash
python main.py --image directory/path/to/image.jpg
```

### Command Line Option

- `--image`: Path to the card image file (JPEG or PNG)


### Example Usage

```bash
# Basic usage
python main.py --image user_input.jpg
