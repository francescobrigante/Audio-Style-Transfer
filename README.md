# Disentangled Latent Representations for Audio Style Transfer

Neural Audio StyleTtransfer between piano and violin using GAN-based Encoder-Decoder architecture based on Disentanglement of Latent Representations of the input in complex domain.

## Overview

This project implements a deep learning system that can transfer the style characteristics between different musical instruments while preserving the musical content. The system uses a combination of STFT and CQT representations (using both real and imaginary parts), 2 transformer encoders: one for style and one for content, an autoregressive decoder and a discriminator to perform adversarial training using also contrastive and reconstruction losses.

## Key Features

- **Operating in the full complex domain**: Many musical models discard phase information, which is essential for a better reconstruction
- **Different encoders for different roles**: Definition of content and style encoder and relative constraints and losses to enforce this
- **Transformer-based Architecture**: Uses self-attention mechanisms for temporal modeling
- **Multi-scale Losses**: Combines time-domain, frequency-domain, and perceptual losses (reconstruction, contrastive, adversarial, disentanglement)
- **Dynamic Sequence Handling**: Supports variable-length audio sequences
- **Curriculum Learning**: Progressive training strategy for stable convergence
- **Memory Efficient**: Optimized for training with limited GPU memory
- **Easy to Scale to more instruments**: Architecture designed to be easy to add more instrument classes

## Architecture

The system consists of four main components:

1. **Style Encoder**: Extracts instrument-specific style embeddings
2. **Content Encoder**: Captures musical content independent of instrument
3. **Autoregressive Decoder**: Reconstructs STFT with transferred style
4. **Discriminator**: Ensures proper disentanglement between style and content

## Repository Structure
```
Audio-Style-Transfer/
├── README.md
│
├── Core Models/
│   ├── style_encoder.py                            # Style encoder with CNN + Transformer
│   ├── content_encoder.py                          # Content encoder for musical structure
│   ├── new_decoder.py                              # Dynamic autoregressive decoder
│   └── discriminator.py                            # Adversarial discriminator
│
├── Training & Loss Functions/
│   ├── losses.py                                   # InfoNCE, margin, adversarial, HSIC losses
│   ├── dataloader.py                               # Efficient dual-instrument data loading
│   ├── train.ipynb                                 # Basic training notebook
│   └── train2.ipynb                                # Advanced training with curriculum learning
│
├── Utilities & Testing/
│   ├── utilityFunctions.py                         # STFT/CQT processing, audio I/O
│   ├── test_correctness.ipynb                      # Model validation and testing
│   └── style_transfer_inference.py                 # Inference script for style transfer
│
├── Dataset Statistics/
│   └── train_set_stats/
│       ├── stats_stft_cqt_piano.npz                # Piano normalization statistics
│       ├── stats_stft_cqt_violin.npz               # Violin normalization statistics
│       └── stats_unified_stft_cqt.npz              # Combined statistics
|
├── Dataset Preprocessing/
│   └── Preprocessing_Dataset/
│       ├── unifies_violin_datasets.py              # Merges Bach and Etudes violin datasets
│       ├── split_BachViolinDataset.py              # Segments Bach violin recordings
│       ├── split_PianoMotion10M.py                 # Extracts piano segments from PianoMotion10M
│       ├── split_ViolinEtudes.py                   # Segments violin etudes recordings
│       ├── compute_unified_stats.py                # Computes combined normalization statistics
│       ├── compute_separated_stats.py              # Computes instrument-specific statistics
│       ├── read_unified_npz.py                     # Utility to inspect unified statistics
│       ├── read_separated_npz.py                   # Utility to inspect separated statistics
│       ├── dataset_trace_analysis.py               # Analyzes audio characteristics and metrics
│       └── dataset_variety.py                      # Visualizes dataset diversity using t-SNE
```


## File Descriptions

### Core Models

#### `style_encoder.py`
Implements the StyleEncoder class using ResNet-like CNN blocks followed by transformer layers. Key features:
- Extracts style embeddings that capture instrument-specific characteristics
- Provides both individual and class-level embeddings (by aggregation through CLS token)
- Uses full STFT + CQT

#### `content_encoder.py`
ContentEncoder for extracting musical content representations. Architecture similar to style encoder, key differences:
- Uses instance normalization instead of batch normalization
- No CLS token to focus on temporal content structure
- Outputs sequence-level embeddings for temporal modeling, perfect for the autoregressive decoder

#### `new_decoder.py`
Dynamic transformer decoder that reconstructs STFT spectrograms. Features:
- **Autoregressive generation** during inference with causal masking
- **Teacher forcing** during training for stable convergence
- **CNN encoder-decoder** for spatial feature processing
- **Dynamic sequence length** handling without fixed parameters
- **Comprehensive loss functions**

#### `discriminator.py`
Simple MLP discriminator for adversarial training:
- Ensures proper disentanglement between style and content representations
- Designed to correctly classify style/class_emb while performing randomly on content_emb

### Training & Loss Functions

#### `losses.py`
Comprehensive loss function library including:
- **InfoNCE loss**: Contrastive learning for style representation
- **Margin loss**: Class separation in embedding space
- **Adversarial losses**: Generator/discriminator training
- **HSIC-based disentanglement loss**: Statistical independence between style and content

#### `dataloader.py`
Efficient data loading for dual-instrument training:
- Handles STFT/CQT computation with configurable parameters
- Applies instrument-specific normalization statistics
- Creates overlapping windows for temporal modeling
- Ensures balanced batch creation across instrument classes for the contrastive training


### Dataset Statistics (`train_set_stats/`)

#### `stats_stft_cqt_piano.npz`
Piano-specific normalization statistics:
- `stft_mean`, `stft_std`: STFT normalization parameters (shape: [2, 513])
- `cqt_mean`, `cqt_std`: CQT normalization parameters (shape: [2, 84])
- Computed from piano training data for instrument-specific normalization

#### `stats_stft_cqt_violin.npz`
Violin-specific normalization statistics:
- Same structure as piano statistics
- Ensures proper normalization for violin characteristics

#### `stats_unified_stft_cqt.npz`
Combined statistics from both instruments:
- Used as fallback when separate statistics are unavailable
- Provides general normalization for mixed-instrument scenarios
- Maintains compatibility with different data configurations

## Training Strategy
The system uses a curriculum learning approach:
- Phase 1 (0-20%): Reconstruction-only training
- Phase 2 (20-40%): Add disentanglement losses
- Phase 3 (40-60%): Introduce contrastive learning
- Phase 4 (60-100%): Full adversarial training
This progressive approach ensures stable training and better convergence.

### Requirements
```
Python 3.8+
PyTorch 1.12+
torchaudio
librosa
numpy
matplotlib
soundfile
```
