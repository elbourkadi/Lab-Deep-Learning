# Lab-Deep-Learning

Clone the repository: git clone https://github.com/elbourkadi/Lab-Deep-Learning.git


This repository includes a project that aimed at building deep learning skills through developing Deep Neural Network (DNN) architectures.


-------

### Objective
To develop hands-on experience in creating and fine-tuning DNN architectures with PyTorch, applying these models to both regression and classification datasets.

-------

### Results
This lab showcases DNNs' potential for both regression and classification and underscores the importance of data preprocessing, hyperparameter tuning, and regularization for model performance.


-----------------------------------------------------------------------------------------------------

## Project 

### Part 1: Regression on NYSE Dataset

1. **Data Preparation**: Load, clean, and split the data into features and target variables.
2. **Exploratory Data Analysis (EDA)**: Visualize trends to inform modeling choices.
3. **DNN Architecture for Regression**: Implement a PyTorch DNN model to predict stock prices.
4. **Hyperparameter Tuning**: Use `GridSearchCV` to optimize parameters.
5. **Model Visualization**: Plot Loss and Accuracy trends for training and test sets.
6. **Regularization**: Add dropout, L1, and L2 regularization to reduce overfitting.

### Part 2: Multi-Class Classification for Predictive Maintenance

1. **Data Preprocessing**: Clean, handle missing values, and normalize the data.
2. **EDA and Data Augmentation**: Perform EDA and augment the data to balance classes.
3. **DNN Architecture for Classification**: Build a DNN model for maintenance type classification.
4. **Hyperparameter Tuning**: Optimize with `GridSearch`.
5. **Evaluation and Visualization**: Visualize Loss and Accuracy, and calculate metrics like F1-score.
6. **Regularization**: Apply dropout and other techniques for better generalization.

------------------------------------------------------------------------------------------------------


# Synthesis: Key Learnings from the Lab

This lab provided practical experience with PyTorch, focusing on developing deep learning models for regression and classification tasks. Key takeaways include the importance of thorough data preparation and exploratory analysis to optimize model inputs, the process of designing and implementing DNNs tailored to different problem types, and the impact of hyperparameter tuning on model accuracy and efficiency. Additionally, applying regularization techniques such as dropout and L1/L2 regularization helped improve model generalization, while evaluating metrics like accuracy and F1 scores, along with visualizing training progress, highlighted the model’s performance and potential areas for improvement.
