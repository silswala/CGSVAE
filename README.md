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

## How to Run a Training Job
Follow this three-step guide to configure and launch a new training experiment.

### Step 1: Configure Your Experiment in the Script

All experiment parameters are defined directly within the create_config.py script. Before running anything, you must first edit this file.

Open create_config.py and modify the config_data dictionary to match your setup:

**Crucial**: Update the data section with the absolute paths to your SQLite database files.

```
# Inside create_config.py
      config_data = {
            # ...
            "data": {
                  "train_db_names": [
                        r"/path/to/your/train.db"
                  ],
                  "test_db_names": [
                        r"/path/to/your/test.db"
                  ],
            },
            # ...
      }
```

**Recommended**: Adjust other parameters as needed, such as `learning_rate`, `epoch`, `batch_size`, and the list of `losses` to use.

### Step 2: Generate the Final Configuration File

After saving your changes to `create_config.py`, run the script from your terminal.

```
python create_config.py
```

This will create a file named `config` inside the `configs/` directory. This file contains the finalized settings from the script and is now ready to be used for training.

### Step 3: Start the Training

Now, you can start the training process using `main.py`. This script reads the settings from your newly generated configs/config file and uses the other command-line arguments to launch the experiment.

You must provide the path to your config file and the `model_type` you wish to train.

**Training Commands**:

* To train the standard CGSVAE for unmixing:

```
python main.py --config_file configs/config --model_type CGSVAE
```

* To train the conditional cCGSVAE for generation:

```
python main.py --config_file configs/config --model_type cCGSVAE
```


* To train the InCGSVAE for inpainting:

```
python main.py --config_file configs/config --model_type InCGSVAE
```

## Output
After running a training job, a new directory will be created under `output/` (or your specified --logdir). The directory will be named with a unique run ID (e.g., 20250826_225303_a1b2c3d4). Inside this directory, you will find:

`model_run_[run_id].log`: A detailed log file of the entire training process.

`config_[run_id].json`: A copy of the exact configuration used for the run.

`output_[run_id].db`: An SQLite database containing the training and testing loss values for each epoch.

`checkpoints/`: A folder containing the saved model weights (.pth files) at specified intervals.

  
