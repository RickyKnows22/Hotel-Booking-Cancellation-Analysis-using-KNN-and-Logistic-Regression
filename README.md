### 1. Import Libraries
- Imports necessary libraries like pandas, numpy, matplotlib, seaborn, ydata_profiling, sklearn modules, and warning.

### 2. Load Dataset
- Reads a CSV file named "booking.csv" into a pandas DataFrame.

### 3. Data Profiling
- Generates a Pandas Profiling Report using the ydata_profiling library, which gives an overview of the dataset including statistics, missing values, and more.

### 4. Understanding the Dataset
- Prints out information about the dataset, such as column data types and non-null counts.

### 5. Summary Statistics
- Provides descriptive statistics for numerical columns in the dataset.

### 6. Check for Unique Values
- Counts unique values in each column.

### 7. Check for Null Values
- Prints the sum of null values for each column.

### 8. Data Cleansing & Preprocessing
- Removes outliers from numerical columns "lead time" and "average price" using the IQR method.
- Displays box plots before and after removing outliers for visualization.
- Encodes categorical variables using one-hot encoding.
- Converts the "booking status" column to binary values (1 for "Canceled", 0 for "Not_Canceled").
- Extracts day, month, and year from the "date of reservation" column and drops the original date column.
- Rounds the "average price" column to integer values.
- Converts boolean columns to integer values (True to 1, False to 0).

### 9. Visualize Correlation
- Creates a heatmap to visualize the correlation between variables.

### 10. featureselection

This function takes a DataFrame `booking` as input and performs the following steps:

- Extracts features and target variables from the DataFrame.
- Uses SelectKBest from sklearn to select the top 10 features based on their scores with respect to the target variable ("booking status").
- Retrieves the indices of the selected features, their scores, and sorts them in descending order.
- Displays the top 10 selected features along with their scores.
- Visualizes the feature importance scores using a horizontal bar chart.
- Returns the selected features (`X`) and target variable (`y`) for further processing.

### 11. training models

This function takes the selected features (`X`) and target variable (`y`) as input and performs the following steps:

- Splits the dataset into training and testing sets using a 80/20 split ratio.
- Trains a Logistic Regression model:
  - Uses GridSearchCV for hyperparameter tuning (parameters `C` and `penalty`).
  - Computes accuracy, confusion matrix, and classification report on the test set.
- Trains a K-Nearest Neighbors (KNN) model:
  - Uses GridSearchCV for hyperparameter tuning (parameter `n_neighbors`).
  - Computes accuracy, confusion matrix, and classification report on the test set.
- Compares the accuracy of both models and concludes that KNN performs better than Logistic Regression.

