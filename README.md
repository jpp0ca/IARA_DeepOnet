# IARA
This project aims to disseminate acoustic data through the implementation of scientific work.
The repository contains PyTorch code for reading the archive in different configurations,
    separating standard folds, and training structures for the results presented in the article.

## Authors
- **Main author**: Fábio Oliveira
- **Advisor**: Natanael Junior

## Repository Structure

- **data/:** This directory serves as the default location to store raw and processed data.
- **notebooks/:** Explore demonstrations and tutorials in Jupyter notebooks.
- **src/:** Contains the code organized as a pip package. It includes functionality for reading data, separating folds, and training models
- **test_scripts/:** Utilize scripts for development and testing of features within the pip package.
- **training_scripts/:** Access scripts used for training, generating plots, and creating tables presented in the final article.

## Installation

### Development Mode
To install the IARA library in development mode, navigate to the `src` folder and run the following command
```bash
pip install -e . --user
```

### Deployment Mode
To install the IARA library in deploy mode, navigate to the `src` folder and run the following command
```bash
pip install .
```


## Usage

Refer to the `notebooks` directory for a detailed guide on accessing and using IARA
