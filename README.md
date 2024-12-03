# The Impact Of Statement Length On Fake News Detection Using Large Langugage Models

In recent months, I have been investigating the impact of statement length on the accuracy of RoBERTa and XLM-R models. This study utilized the LIAR2 dataset and involved a total of 12 experiments. The tests were conducted using the original statements, expanded statements with various feature sets, and both RoBERTa and XLM-R models.

The research was conducted on the GPU4EDU server at Tilburg University. Utilizing this server allowed for efficient processing and experimentation, but it also required the project files to be organized and divided into specific components. These included:
-Data Cleaning and Exploratory Data Analysis (EDA): This stage involved preparing the data by cleaning and preprocessing it, followed by conducting an exploratory analysis to understand key patterns and distributions within the dataset.
-Experimentation Files: Examples of experiments conducted include testing various model setups, analyzing the impact of different features, and evaluating performance across diverse scenarios.
-Optuna Hyperparameter Optimization: Experiments were carried out using Optuna to fine-tune the model hyperparameters, aiming to achieve optimal performance. This process involved testing various configurations and selecting the best-performing settings.
-LIME Analysis: Local Interpretable Model-Agnostic Explanations (LIME) were used to interpret the model’s predictions, providing insights into which features contributed most significantly to specific outcomes.
Each of these components played a crucial role in ensuring the robustness and interpretability of the research findings, leveraging the computational capabilities of GPU4EDU to handle complex and resource-intensive tasks efficiently.


