# Machine Learning and Data Analysis for GW Detection

This repository contains a series of Jupyter Notebooks that implement machine learning models and data analysis techniques to process and analyze gravitational wave (GW) detection data. Below is a description of each file to guide users through the content and its purpose.

## Files and Descriptions

### Data Analysis and Preprocessing
- **`Analysis_data.ipynb`**  
  Notebook for analyzing the initial dataset, including basic preprocessing, feature exploration, and summary statistics.

- **`Combine_ChripMass.ipynb`**  
  Combines different pipelines into different bins of chirp mass data.

- **`LUCAS_read_found_inj_full_data_trigger_info.ipynb`**  
  Reads and processes detailed trigger information, focusing on found background for data exploration.

- **`Newdata.ipynb`**  
  Generates and processes a new dataset to enhance background information for machine learning models.

- **`read_found_inj_full_data_trigger_info.ipynb`**  
  Focuses on reading and summarizing data from found injections to extract relevant trigger parameters.

### Machine Learning Models
- **`DecisionTreesRegresor.ipynb`**  
  Implements a Decision Tree Regressor to predict physical parameters of GW signals based on the dataset.

- **`LogisticRegressionClassifier.ipynb`**  
  Uses Logistic Regression for binary classification of GW events.

- **`ML_combine_allPipeline.ipynb`**  
  Combines all pipelines for machine learning models to streamline data preprocessing, training, and evaluation.

- **`RandomForestClassifier.ipynb`**  
  Implements a Random Forest Classifier to categorize GW events and evaluate performance metrics.

- **`RandomForestRegressor.ipynb`**  
  Builds a Random Forest Regressor to predict continuous variables such as IFAR.

- **`New_Background_DecisionTreesRegresor.ipynb`**  
  Applies a Decision Tree Regressor using new background data to improve model performance.

- **`New_Background_RandomForestRegressor.ipynb`**  
  Tests a Random Forest Regressor with additional background data for enhanced predictions.

### Advanced Analyses
- **`different_analysis_ML.ipynb`**  
  Explores various machine learning approaches to analyze GW data, including comparisons of model performance.

- **`combine_pipelines.ipynb`**  
  Integrates multiple data processing pipelines for an end-to-end workflow in GW signal detection.

### Auxiliary Files
- **`Add files via upload`**  
  Placeholder description for recently uploaded files.

## Usage
Each notebook is designed to perform specific tasks in data analysis or machine learning. To get started:
1. Clone this repository.
2. Open the notebooks in a Jupyter Notebook environment.
3. Follow the instructions within each notebook to replicate or extend the analyses.

## Prerequisites
Ensure the following dependencies are installed:
- Python 3.8+
- Jupyter Notebook
- Machine learning libraries: `scikit-learn`, `numpy`, `pandas`
- Visualization libraries: `matplotlib`, `seaborn`
