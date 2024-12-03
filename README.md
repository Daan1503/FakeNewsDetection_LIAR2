# The Impact Of Statement Length On Fake News Detection Using Large Langugage Models

In recent months, I have been investigating the impact of statement length on the accuracy of RoBERTa and XLM-R models. This study utilized the LIAR2 dataset and involved a total of 12 experiments. The tests were conducted using the original statements, expanded statements with various feature sets, and both RoBERTa and XLM-R models.

The research was carried out on the GPU4EDU server provided by Tilburg University. Leveraging this server enabled efficient processing and experimentation. To effectively organize the project, the files were divided into several key components and to make it easier to run on the GPU4EDU server:
### 1. Data Cleaning and Exploratory Data Analysis (EDA)
  - Data cleaning to remove inconsistencies and errors.
  - Exploratory Data Analysis (EDA) to identify key patterns and trends within the dataset.
  - 
### 2. Statement expanding with Llama 3.1
  - The code for text expanding the statement with minimal of 20 tokens with the Llama 3.1 8B parameters LLM.

### 3. Optuna Hyperparameter Optimization
  -Used Optuna to automate the hyperparameter search.

### 4. Experiment 
  - One example of the test that where performed during the research.

### 5. LIME Analysis
  - Gained insights into feature importance of Top Words per class.
  - Improved understanding of how specific words influence the model's decisions.


