# The Impact Of Statement Length On Fake News Detection Using Large Langugage Models

In recent months, I have been investigating the impact of statement length on the accuracy of RoBERTa and XLM-R models. This study utilized the LIAR2 dataset and involved a total of 12 experiments. The tests were conducted using the original statements, expanded statements with various feature sets, and both RoBERTa and XLM-R models.


The research was carried out on the GPU4EDU server provided by Tilburg University. Leveraging this server enabled efficient processing and experimentation. To organize the project effectively, the files were divided into several key components:
## 1. Data Cleaning and Exploratory Data Analysis (EDA)
- **Objective:** Prepare and preprocess the data to ensure it was clean and ready for analysis.
- **Steps:**
  - Data cleaning to remove inconsistencies and errors.
  - Exploratory Data Analysis (EDA) to identify key patterns and trends within the dataset.

## 2. Experimentation Files
- **Examples of Experiments:**
  - Testing various model configurations and feature sets.
  - Evaluating performance under different experimental conditions.

## 3. Optuna Hyperparameter Optimization
- **Objective:** Fine-tune model hyperparameters to maximize performance.
- **Methodology:**
  - Used Optuna to automate the hyperparameter search.
  - Experimented with multiple configurations to identify the optimal setup.

## 4. LIME Analysis
- **Objective:** Interpret model predictions using Local Interpretable Model-Agnostic Explanations (LIME).
- **Outcome:** 
  - Gained insights into feature importance.
  - Improved understanding of how specific features influenced the model's decisions.


