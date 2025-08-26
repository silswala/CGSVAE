# CGSVAE: A Deep Learning Toolkit for Gamma Spectrum Analysis
This repository contains a PyTorch-based toolkit for the quantitative analysis, conditional generation, and inpainting of low-resolution Plastic Scintillator(PS) gamma-ray spectra using Composition-Guided Variational Autoencoders (VAEs).

## Overview
Working with low-resolution plastic scintillator detector spectrum presents a challenge: The spectrum often broad and featureless, making it difficult to determine which radioactive sources are present and in what amounts. This toolkit provides a flexible framework and three specialized VAE models to address this problem by learning the underlying composition of complex spectra.

## Features
1. Three Specialized Models: Includes implementations for standard, conditional, and inpainting VAEs.

2. Configuration-Driven: All experiment parameters are managed through a central configuration, making runs easily reproducible.

3. Customizable Training: Supports a variety of loss functions that can be combined to suit specific training objectives.

4. Robust Data Handling: Natively loads data from SQLite databases for efficient management of large datasets.

5. Comprehensive Logging: Automatically saves detailed logs, model checkpoints, and a database of loss metrics for every run.

## Implemented Models
This framework includes three distinct models, each selectable via a command-line argument.

### CGSVAE (Composition-Guided Spectrum VAE)

The core model for quantitative unmixing. Its decoder has two output heads: one to reconstruct the input spectrum and another to predict its underlying radionuclide composition. This "composition guidance" is key to its high performance.

### cCGSVAE (Conditional CGS-VAE)

A conditional VAE designed for on-demand spectrum generation. Both the encoder and decoder take a radionuclide composition vector as an additional input, allowing the model to generate novel spectra that match specific, user-defined conditions.

### InCGSVAE (Inpainting CGS-VAE)

A model trained to reconstruct complete, clean spectra from corrupted or incomplete inputs. During training, input data is artificially masked, and the model learns to "inpaint" the missing regions by leveraging its understanding of valid spectral shapes and compositions.

## Data Preparation
The training and testing scripts expect data to be stored in SQLite database (.db) files. Each database must contain a table with the following structure:
      
  * Table Name: ps_spectra

  * Columns:

      - ps_spectrum: A JSON string representing the spectrum array (e.g., a list of 1024 floats).

      - radionuclide_percentage: A JSON string representing the corresponding label array (e.g., a list of 6 floats).

The data_loader.py script will handle reading these JSON strings, converting them to NumPy arrays, and performing min-max normalization on the spectra.

## 


  
