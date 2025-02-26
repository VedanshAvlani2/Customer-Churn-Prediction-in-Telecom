# Load necessary libraries
if (!require(caret)) install.packages("caret")
library(caret)

if (!require(randomForest)) install.packages("randomForest")
library(randomForest)

if (!require(rpart)) install.packages("rpart")
library(rpart)

if (!require(rpart.plot)) install.packages("rpart.plot")
library(rpart.plot)

if (!require(neuralnet)) install.packages("neuralnet")
library(neuralnet)

if (!require(ROSE)) install.packages("ROSE")
library(ROSE)

if (!require(dplyr)) install.packages("dplyr")
library(dplyr)

if (!require(ggplot2)) install.packages("ggplot2")
library(ggplot2)

library(corrplot)

# Load the dataset
data <- read.csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Data Preprocessing
data$TotalCharges <- as.numeric(as.character(data$TotalCharges))
data <- na.omit(data)

# Convert relevant variables to factors
data$SeniorCitizen <- as.factor(data$SeniorCitizen)
data$Churn <- as.factor(data$Churn)
data$gender <- as.factor(data$gender)
data$Partner <- as.factor(data$Partner)
data$Dependents <- as.factor(data$Dependents)
data$PhoneService <- as.factor(data$PhoneService)
data$MultipleLines <- as.factor(data$MultipleLines)
data$InternetService <- as.factor(data$InternetService)
data$OnlineSecurity <- as.factor(data$OnlineSecurity)
data$TechSupport <- as.factor(data$TechSupport)
data$Contract <- as.factor(data$Contract)
data$PaperlessBilling <- as.factor(data$PaperlessBilling)
data$PaymentMethod <- as.factor(data$PaymentMethod)

# Remove non-useful variables if any (e.g., IDs)
data$customerID <- NULL  # Remove if present

# Train-Test Split
set.seed(123)
trainIndex <- sample(1:nrow(data), 0.8 * nrow(data))
trainData <- data[trainIndex, ]
testData <- data[-trainIndex, ]

# Preprocess training data for ROSE
preprocess_data_for_rose <- function(data) {
  # Ensure all categorical variables are factors
  categorical_vars <- c('gender', 'Partner', 'Dependents', 'PhoneService', 
                        'MultipleLines', 'InternetService', 'OnlineSecurity',
                        'OnlineBackup', 'DeviceProtection', 'TechSupport',
                        'StreamingTV', 'StreamingMovies', 'Contract',
                        'PaperlessBilling', 'PaymentMethod', 'Churn')
  
  for (var in categorical_vars) {
    if (var %in% names(data)) {
      data[[var]] <- as.factor(data[[var]])
    }
  }
  
  # Ensure numeric variables are properly typed
  numeric_vars <- c('tenure', 'MonthlyCharges', 'TotalCharges')
  
  for (var in numeric_vars) {
    if (var %in% names(data)) {
      data[[var]] <- as.numeric(as.character(data[[var]]))
    }
  }
  
  return(data)
}

# Preprocess training data for ROSE
trainData <- preprocess_data_for_rose(trainData)

# Apply ROSE to balance the dataset
rose_data <- ROSE(Churn ~ ., data = trainData, seed = 123)$data

# Check the class distribution after applying ROSE
table(rose_data$Churn)

# Logistic Regression Model
log_model <- glm(Churn ~ ., data = rose_data, family = binomial)

# Logistic Regression Predictions
log_pred_probs <- predict(log_model, testData, type = "response")
log_pred_class <- ifelse(log_pred_probs > 0.5, "Yes", "No")
log_pred_class <- factor(log_pred_class, levels = levels(testData$Churn))

# Evaluate Logistic Regression
log_conf_matrix <- table(log_pred_class, testData$Churn)
log_accuracy <- sum(diag(log_conf_matrix)) / sum(log_conf_matrix)

print("Logistic Regression Confusion Matrix:")
print(log_conf_matrix)
print(paste("Logistic Regression Accuracy:", round(log_accuracy, 4)))

# Random Forest Model
set.seed(123)
rf_model <- randomForest(Churn ~ ., 
                         data = rose_data, 
                         ntree = 500,              # Number of trees
                         mtry = floor(sqrt(ncol(rose_data) - 1)), # Number of features per split
                         importance = TRUE,        # Measure feature importance
                         nodesize = 5)             # Minimum size of terminal nodes

# Align Factor Levels Between rose_data and testData
for (var in colnames(rose_data)) {
  if (is.factor(rose_data[[var]]) && var %in% colnames(testData)) {
    testData[[var]] <- factor(testData[[var]], levels = levels(rose_data[[var]]))
  }
}

# Random Forest Predictions
rf_pred_probs <- predict(rf_model, testData, type = "prob")
rf_pred_class <- predict(rf_model, testData, type = "response")

# Evaluate Random Forest Model
rf_conf_matrix <- confusionMatrix(rf_pred_class, testData$Churn)
rf_accuracy <- rf_conf_matrix$overall["Accuracy"]

print("Random Forest Results:")
print(rf_conf_matrix)
print(paste("Random Forest Accuracy:", round(rf_accuracy, 4)))

# Feature Importance Plot
print("Feature Importance from Random Forest:")
varImpPlot(rf_model, main = "Feature Importance Plot")

# Decision Tree Model
set.seed(123)
dt_model <- rpart(Churn ~ ., data = rose_data, method = "class", 
                  control = rpart.control(cp = 0.005, minsplit = 15, maxdepth = 8))

# Visualize the Decision Tree
rpart.plot(dt_model, main = "Decision Tree for Churn Prediction")

# Decision Tree Predictions
dt_pred_probs <- predict(dt_model, testData, type = "prob")
dt_pred_class <- ifelse(dt_pred_probs[, 2] > 0.5, "Yes", "No")
dt_pred_class <- factor(dt_pred_class, levels = levels(testData$Churn))

# Evaluate Decision Tree
dt_conf_matrix <- confusionMatrix(dt_pred_class, testData$Churn)
dt_accuracy <- dt_conf_matrix$overall["Accuracy"]

print("Decision Tree Results:")
print(dt_conf_matrix)
print(paste("Decision Tree Accuracy:", round(dt_accuracy, 4)))

# Model Comparison
model_comparison <- data.frame(
  Model = c("Random Forest", "Decision Tree"),
  Accuracy = c(rf_accuracy, dt_accuracy)
)

print("Model Comparison:")
print(model_comparison)

# Neural Network Model
# Scale numeric variables
scaled_train <- rose_data
scaled_test <- testData

normalize <- function(x) {
  return((x - min(x)) / (max(x) - min(x)))
}

numeric_vars <- c("tenure", "MonthlyCharges", "TotalCharges")
scaled_train[, numeric_vars] <- lapply(scaled_train[, numeric_vars], normalize)
scaled_test[, numeric_vars] <- lapply(scaled_test[, numeric_vars], normalize)

# Encode Churn as numeric
scaled_train$Churn_numeric <- ifelse(scaled_train$Churn == "Yes", 1, 0)
scaled_test$Churn_numeric <- ifelse(scaled_test$Churn == "Yes", 1, 0)

# One-hot encode categorical variables for neural network
one_hot_train <- model.matrix(~ gender + SeniorCitizen + Partner + Dependents + 
                                PhoneService + MultipleLines + InternetService + 
                                OnlineSecurity + TechSupport + Contract + 
                                PaperlessBilling + PaymentMethod - 1, data = scaled_train)
one_hot_test <- model.matrix(~ gender + SeniorCitizen + Partner + Dependents + 
                               PhoneService + MultipleLines + InternetService + 
                               OnlineSecurity + TechSupport + Contract + 
                               PaperlessBilling + PaymentMethod - 1, data = scaled_test)

# Combine numeric and encoded data
nn_train <- cbind(one_hot_train, scaled_train[, c("tenure", "MonthlyCharges", "TotalCharges", "Churn_numeric")])
nn_test <- cbind(one_hot_test, scaled_test[, c("tenure", "MonthlyCharges", "TotalCharges", "Churn_numeric")])

# Sanitize column names
colnames(nn_train) <- make.names(colnames(nn_train))
colnames(nn_test) <- make.names(colnames(nn_test))

# Extract sanitized predictor names
predictor_names <- setdiff(names(nn_train), "Churn_numeric")

# Create formula dynamically
nn_formula <- as.formula(paste("Churn_numeric ~", paste(predictor_names, collapse = " + ")))

# Train Neural Network
nn_model <- neuralnet(nn_formula, data = nn_train, hidden = c(3), linear.output = FALSE, stepmax = 1e6)

# Check if the model converged
if (!is.null(nn_model$weights)) {
  # Visualize Neural Network
  plot(nn_model, main = "Neural Network for Churn Prediction")
  
  # Neural Network Predictions
  nn_pred_probs <- neuralnet::compute(nn_model, nn_test[, predictor_names])$net.result
  
  # Convert probabilities to class predictions
  nn_pred_class <- ifelse(nn_pred_probs > 0.5, "Yes", "No")
  nn_pred_class <- factor(nn_pred_class, levels = levels(testData$Churn))
  
  # Evaluate Neural Network
  nn_conf_matrix <- table(nn_pred_class, testData$Churn)
  nn_accuracy <- sum(diag(nn_conf_matrix)) / sum(nn_conf_matrix)
  
  print("Neural Network Confusion Matrix:")
  print(nn_conf_matrix)
  print(paste("Neural Network Accuracy:", round(nn_accuracy, 4)))
} else {
  print("Neural network did not converge. Try increasing stepmax or simplifying the architecture.")
}

# Model Comparison
model_comparison <- data.frame(
  Model = c("Logistic Regression", "Decision Tree", "Random Forest", "Neural Network"),
  Accuracy = c(round(log_accuracy, 4), round(dt_accuracy, 4), round(rf_accuracy, 4), round(nn_accuracy, 4))
)

print("Model Comparison:")
print(model_comparison) 

# Summary Statistics
summary(data)

# Distribution of Tenure
ggplot(data, aes(x = tenure)) +
  geom_histogram(binwidth = 5, fill = "lightblue", color = "black") +
  labs(title = "Distribution of Tenure", x = "Tenure (Months)", y = "Count")

# Distribution of Monthly Charges
ggplot(data, aes(x = MonthlyCharges)) +
  geom_histogram(binwidth = 5, fill = "lightgreen", color = "black") +
  labs(title = "Distribution of Monthly Charges", x = "Monthly Charges", y = "Count")

# Distribution of Total Charges
ggplot(data, aes(x = TotalCharges)) +
  geom_histogram(binwidth = 100, fill = "lightcoral", color = "black") +
  labs(title = "Distribution of Total Charges", x = "Total Charges", y = "Count")

# Churn Distribution
ggplot(data, aes(x = Churn)) +
  geom_bar(fill = "steelblue") +
  labs(title = "Churn Distribution", x = "Churn", y = "Count")

# Gender Distribution
ggplot(data, aes(x = gender)) +
  geom_bar(fill = "lightblue") +
  labs(title = "Gender Distribution", x = "Gender", y = "Count")

# Contract Type Distribution
ggplot(data, aes(x = Contract)) +
  geom_bar(fill = "orange") +
  labs(title = "Contract Types Distribution", x = "Contract Type", y = "Count")

# Churn Rate by Contract Type
ggplot(data, aes(x = Contract, fill = Churn)) +
  geom_bar(position = "fill") +
  labs(title = "Churn Rate by Contract Type", x = "Contract Type", y = "Proportion") +
  scale_y_continuous(labels = scales::percent)

# Monthly Charges by Churn
ggplot(data, aes(x = Churn, y = MonthlyCharges, fill = Churn)) +
  geom_boxplot() +
  labs(title = "Monthly Charges by Churn", x = "Churn", y = "Monthly Charges")

# Total Charges by Churn
ggplot(data, aes(x = Churn, y = TotalCharges, fill = Churn)) +
  geom_boxplot() +
  labs(title = "Total Charges by Churn", x = "Churn", y = "Total Charges")

# K-means Clustering Visualization
numeric_features <- scale(data %>% select(tenure, MonthlyCharges, TotalCharges))
set.seed(123)
kmeans_result <- kmeans(numeric_features, centers = 3, nstart = 25)
data$Cluster <- as.factor(kmeans_result$cluster)
ggplot(data, aes(x = MonthlyCharges, y = TotalCharges, color = Cluster)) +
  geom_point(alpha = 0.6) +
  labs(title = "K-means Clustering", x = "Monthly Charges", y = "Total Charges")

# Correlation Matrix Visualization
cor_matrix <- cor(data %>% select(tenure, MonthlyCharges, TotalCharges), use = "complete.obs")
corrplot(cor_matrix, method = "circle", main = "Correlation Matrix")

# ROC Curve for Random Forest Model
if (exists("results")) {
  roc_obj <- roc(results$test_data$Churn, results$predictions[, 2])
  plot(roc_obj, main = "ROC Curve", col = "blue")
}

# Class Distribution in Balanced Dataset
if (exists("results")) {
  barplot(table(results$balanced_data$Churn),
          main = "Class Distribution after Balancing",
          col = c("skyblue", "coral"))
}

# Numeric Variables Distribution in Balanced Dataset
if (exists("results")) {
  for (var in numeric_vars) {
    hist(results$balanced_data[[var]],
         main = paste("Distribution of", var),
         xlab = var,
         col = "lightgreen")
  }
}

# Variable Importance Plot
if (exists("results")) {
  varImpPlot(results$model, main = "Variable Importance", n.var = min(10, ncol(data) - 1))
}

# Final Model Comparison Visualization
if (exists("model_comparison")) {
  ggplot(model_comparison, aes(x = Model, y = Accuracy, fill = Model)) +
    geom_bar(stat = "identity", color = "black") +
    labs(title = "Model Accuracy Comparison", x = "Model", y = "Accuracy") +
    theme_minimal()
}