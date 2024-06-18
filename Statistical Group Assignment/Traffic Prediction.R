# Load necessary libraries
library(ggplot2)
library(dplyr)
library(class)
library(neuralnet)
library(caret)
library(randomForest)
library(corrplot)
library(e1071)
library(tidyr)
library(grid) 

# Load the data set
traffic_data <- read.csv('Traffic.csv')
head(traffic_data)
str(traffic_data)


# Data Pre processing and Exploration
# Convert 'Time' to a more usable format and extract features
traffic_data$Time <- as.POSIXct(traffic_data$Time, format="%I:%M:%S %p")
traffic_data$Hour <- as.integer(format(traffic_data$Time, "%H"))

traffic_data$Date <- as.Date(traffic_data$Date, format="%d")

# Check for missing values
sum(is.na(traffic_data))

# Summary statistics
summary(traffic_data)

# Data Visualization
# Distribution of Numerical Features

numeric_features <- traffic_data %>% select(CarCount, BikeCount, BusCount, TruckCount, Total)
numeric_features_melted <- numeric_features %>% pivot_longer(cols = everything(), names_to = "variable", values_to = "value")

ggplot(numeric_features_melted, aes(x=value, fill=variable)) +
  geom_histogram(bins=30, alpha=0.7, position="dodge") +
  facet_wrap(~variable, scales="free") +
  theme_minimal() +
  labs(title="Distribution of Numerical Features", x="Value", y="Frequency")

# correlation Metrics
cor_matrix <- cor(numeric_features)
corrplot(cor_matrix, method="color", type="lower", order="hclust", 
         addCoef.col = "black", tl.cex = 0.8, tl.col = "black", 
         cl.pos = "n")

# Relationship between Features and Target Variable
ggplot(traffic_data, aes(x=CarCount, y=Total)) + 
  geom_point(alpha=0.6) + 
  theme_minimal() +
  labs(title="Total Traffic vs Car Count", x="Car Count", y="Total Traffic")

ggplot(traffic_data, aes(x=BikeCount, y=Total)) + 
  geom_point(alpha=0.6) + 
  theme_minimal() +
  labs(title="Total Traffic vs Bike Count", x="Bike Count", y="Total Traffic")

ggplot(traffic_data, aes(x=BusCount, y=Total)) + 
  geom_point(alpha=0.6) + 
  theme_minimal() +
  labs(title="Total Traffic vs Bus Count", x="Bus Count", y="Total Traffic")

ggplot(traffic_data, aes(x=TruckCount, y=Total)) + 
  geom_point(alpha=0.6) + 
  theme_minimal() +
  labs(title="Total Traffic vs Truck Count", x="Truck Count", y="Total Traffic")

# Traffic counts by hour
ggplot(traffic_data, aes(x=Hour, y=Total, fill=Traffic.Situation)) +
  geom_bar(stat="identity") +
  theme_minimal() +
  labs(title="Total Traffic by Hour", x="Hour", y="Total Traffic")


# Traffic counts by day of the week
ggplot(traffic_data, aes(x=Day.of.the.week, y=Total, fill=Traffic.Situation)) +
  geom_bar(stat="identity") +
  theme_minimal() +
  labs(title="Total Traffic by Day of the Week", x="Day of the Week", y="Total Traffic")


# Feature Engineering
# Convert categorical variables to factors
traffic_data$Day.of.the.week <- as.factor(traffic_data$Day.of.the.week)
traffic_data$Hour <- as.factor(traffic_data$Hour)
traffic_data$Traffic.Situation <- as.factor(traffic_data$Traffic.Situation)

# Model Selection and Training
# Split the data into training and testing sets
set.seed(123)
trainIndex <- createDataPartition(traffic_data$Total, p = .8, 
                                  list = FALSE, 
                                  times = 1)
traffic_train <- traffic_data[ trainIndex,]
traffic_test  <- traffic_data[-trainIndex,]

# Define training control
train_control <- trainControl(method="cv", number=10)



# Hyperparameter tuning for Random Forest
rf_grid <- expand.grid(mtry = seq(1, ncol(traffic_train) - 1, by = 1))

# Train Random Forest model
set.seed(123)
rf_model <- train(Total ~ CarCount + BikeCount + BusCount + TruckCount + Day.of.the.week + Hour, 
                  data=traffic_train, 
                  method="rf", 
                  trControl=train_control)
summary(rf_model)


# Predictions and evaluation for Random Forest
rf_predictions <- predict(rf_model, traffic_test)
rf_rmse <- sqrt(mean((rf_predictions - traffic_test$Total)^2))
rf_mae <- mean(abs(rf_predictions - traffic_test$Total))
rf_r2 <- 1 - sum((rf_predictions - traffic_test$Total)^2) / sum((mean(traffic_test$Total) - traffic_test$Total)^2)
print(paste("Random Forest RMSE:", rf_rmse))
print(paste("Random Forest MAE:", rf_mae))
print(paste("Random Forest R²:", rf_r2))

# Train Linear Regression model
#No hyperparameters to tune
lm_model <- lm(Total ~ CarCount + BikeCount + BusCount + TruckCount + Day.of.the.week + Hour, data = traffic_train)
summary(lm_model)

# Predictions and evaluation for Linear Regression
lm_predictions <- predict(lm_model, traffic_test)
lm_rmse <- sqrt(mean((lm_predictions - traffic_test$Total)^2))
lm_mae <- mean(abs(lm_predictions - traffic_test$Total))
lm_r2 <- 1 - sum((lm_predictions - traffic_test$Total)^2) / sum((mean(traffic_test$Total) - traffic_test$Total)^2)
print(paste("Linear Regression RMSE:", lm_rmse))
print(paste("Linear Regression MAE:", lm_mae))
print(paste("Linear Regression R²:", lm_r2))

# Hyperparameter tuning for KNN
knn_grid <- expand.grid(k = seq(1, 20, by = 1))

# Train KNN model
knn_model <- train(Total ~ CarCount + BikeCount + BusCount + TruckCount + Day.of.the.week + Hour, 
                   data=traffic_train, 
                   method="knn", 
                   trControl=train_control,
                   preProcess = c("center", "scale"))
summary(knn_model)

# Predictions and evaluation for KNN
knn_predictions <- predict(knn_model, traffic_test)
knn_rmse <- sqrt(mean((knn_predictions - traffic_test$Total)^2))
knn_mae <- mean(abs(knn_predictions - traffic_test$Total))
knn_r2 <- 1 - sum((knn_predictions - traffic_test$Total)^2) / sum((mean(traffic_test$Total) - traffic_test$Total)^2)
print(paste("KNN RMSE:", knn_rmse))
print(paste("KNN MAE:", knn_mae))
print(paste("KNN R²:", knn_r2))

# Compare the performance metrics of the models
performance_results <- data.frame(
  Model = c("Random Forest", "Linear Regression", "KNN"),
  RMSE = c(rf_rmse, lm_rmse, knn_rmse),
  MAE = c(rf_mae, lm_mae, knn_mae),
  R2 = c(rf_r2, lm_r2, knn_r2)
)

# Plot the comparison of performance metrics
performance_results_melted <- performance_results %>%
  pivot_longer(cols = -Model, names_to = "Metric", values_to = "Value")

ggplot(performance_results_melted, aes(x=Model, y=Value, fill=Model)) +
  geom_bar(stat="identity", position=position_dodge()) +
  facet_wrap(~Metric, scales="free_y") +
  theme_minimal() +
  labs(title="Comparison of Model Performance Metrics", x="Model", y="Value")

