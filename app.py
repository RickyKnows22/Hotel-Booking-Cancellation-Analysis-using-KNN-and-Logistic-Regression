import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def load_data(file_path):
    booking = pd.read_csv(file_path)
    booking.drop(["Booking_ID"], axis=1, inplace=True)
    booking.index = booking.index + 1
    return booking

def preprocess_data(booking):
    st.header("3. Data Preprocessing")
    st.write("Original Dataset:")
    st.write(booking.head())

    st.caption("Data Profiling")
    profile = ProfileReport(booking, title="Pandas Profiling Report")
    st.write("> Generates a Pandas Profiling Report using the ydata_profiling library, which gives an overview of the dataset including statistics, missing values, and more.")

    st.write("Understanding the Dataset:")
    st.write(booking.info())
    st.write("> Prints out information about the dataset, such as column data types and non-null counts.")

    st.write("Summary Statistics of the dataset:")
    st.write(booking.describe())
    st.write("> Provides descriptive statistics for numerical columns in the dataset.")

    st.write("Check for Unique Values:")
    st.write(booking.nunique())
    st.write("> Counts unique values in each column")

    st.write("Check for null values:")
    st.write(booking.isnull().sum().sort_values(ascending=False))
    st.write("> prints the sum of null values for each column")

    st.subheader("Data Cleansing & Preprocessing steps: ")
    st.write('> Remove outliers from numerical columns "lead time" and "average price" using the IQR method.')
    st.write("> Display box plots before and after removing outliers for visualization.")
    st.write("> Encode categorical variables using one-hot encoding.")
    st.write('> Convert the "booking status" column to binary values (1 for "Canceled", 0 for "Not_Canceled").')
    st.write('> Extract day, month, and year from the "date of reservation" column and drop the original date column.')
    st.write('> Round the "average price" column to integer values.')
    st.write("> Convert boolean columns to integer values (True to 1, False to 0).")
    
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.write("Create box plots for every variable before droping outliers")
    plt.figure(figsize=(12, 8))
    sns.set(style="darkgrid")
    sns.boxplot(data=booking, orient="h")
    plt.title("Box Plot for Every Variable")
    st.pyplot()  # Display the plot in Streamlit
    st.write(booking.shape)


    # Drop outliers
    outliers_cols = ["lead time", "average price"]
    for column in outliers_cols:
        if booking[column].dtype in ["int64", "float64"]:
            q1 = booking[column].quantile(0.25)
            q3 = booking[column].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            booking = booking[
                (booking[column] >= lower_bound) & (booking[column] <= upper_bound)
            ]

    st.write("Create box plots for every variable after dropping outliers")
    plt.figure(figsize=(12, 8))
    sns.set(style="darkgrid")
    booking_boxplot = sns.boxplot(data=booking, orient="h")
    plt.title("Box Plot for Every Variable")
    st.pyplot()
    st.write(booking.shape)
    
    st.write("Dataset after dropping outliers:")
    st.write(booking.head())


    st.subheader("Encoding categorical variables")
    st.write("1. Binarizing the target column {Cancelled = 1, Not_Canceled = 0}")
    booking["booking status"] = booking["booking status"].replace("Canceled", 1)
    booking["booking status"] = booking["booking status"].replace("Not_Canceled", 0)
    st.write(booking.head())

    st.write("2. Split the date and drop the date in the wrong format")
    booking = booking[~booking["date of reservation"].str.contains("-")]
    booking["date of reservation"] = pd.to_datetime(booking["date of reservation"])
    booking["day"] = booking["date of reservation"].dt.day
    booking["month"] = booking["date of reservation"].dt.month
    booking["year"] = booking["date of reservation"].dt.year
   
    st.write("3. Drop the original datetime column")
    booking = booking.drop(columns=["date of reservation"])
    # st.write(booking.info())

    st.write("4. Round the float col to int")
    booking["average price"] = booking["average price"].round().astype(int)

    st.write("5. Apply One Hot Encoding on Variables of datatype = 'object'")
    object_columns = booking.select_dtypes(include=["object"]).columns
    booking = pd.get_dummies(booking, columns=object_columns)
    booking = booking.replace({True: 1, False: 0})
    # st.write(booking.info())
    st.write("Dataset after preprocessing:")
    st.write(booking.head())

    st.subheader("Correlation between variables:")
    st.write("heatmap to visualize the correlation between variables.")
    plt.figure(figsize=(12, 8))
    sns.heatmap(booking.corr(), cmap="icefire", linewidths=0.5)
    plt.title("Correlation Heatmap")
    st.pyplot()

    return booking

def feature_selection(booking):
    st.header("4. Feature Selection")
    features = booking.drop(["booking status"], axis=1)
    target = booking["booking status"]

    k_best = SelectKBest(score_func=f_classif, k=10)
    X = k_best.fit_transform(features, target)
    y = target
    st.write("1. Get the indices of the selected features")
    selected_features_indices = k_best.get_support(indices=True)

    st.write("2. Get the scores associated with each feature")
    feature_scores = k_best.scores_

    st.write("3. Create a list of tuples containing feature names and scores")
    feature_info = list(zip(features.columns, feature_scores))

    st.write("4. Sort the feature info in descending order based on scores")
    sorted_feature_info = sorted(feature_info, key=lambda x: x[1], reverse=True)

    st.subheader("Top 10 Selected Features:")
    for idx, (feature_name, feature_score) in enumerate(sorted_feature_info[:10], start=1):
        st.write(f"{idx}. {feature_name}: {feature_score:.2f}")


    #visualize the relation between features with their score and the target variable
    feature_names, feature_scores = zip(*sorted_feature_info[:])

    # Create a bar chart
    st.write("Bar chart of Feature Importance Scores:")
    plt.figure(figsize=(10, 6))
    plt.barh(feature_names, feature_scores, color="skyblue")
    plt.xlabel("Feature Importance Score")
    plt.title("Features Importance Scores")
    st.pyplot()



    selected_features_df = features.iloc[:, selected_features_indices]
    st.write("Selected Features:")
    st.write(selected_features_df.head())
    return X,y

def train_models(X, y):
    st.header("5. Model Training and Evaluation")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=5
    )

    # Logistic Regression
    st.subheader("Logistic Regression:")
    log_reg = LogisticRegression()
    params = {"C": [0.01, 0.1, 1, 10, 100], "penalty": ["l1", "l2"]}
    grid_search = GridSearchCV(log_reg, param_grid=params, cv=5)
    grid_search.fit(X_train, y_train)
    best_log_reg = grid_search.best_estimator_
    y_pred_lr = best_log_reg.predict(X_test)
    accuracy_lr = accuracy_score(y_test, y_pred_lr)
    st.write(f"Accuracy: {accuracy_lr:.2f}")
    st.write("Best Parameters:", grid_search.best_params_)
    st.write("Best Score:", grid_search.best_score_)
    st.write("Confusion Matrix:")
    st.write(confusion_matrix(y_test, y_pred_lr))
    # st.write("Classification Report:")
    # st.write(classification_report(y_test, y_pred_lr))


    # KNN
    st.subheader("K-Nearest Neighbors (KNN):")
    knn = KNeighborsClassifier()
    params = {"n_neighbors": np.arange(1, 10)}
    grid_search = GridSearchCV(knn, param_grid=params, cv=5)
    grid_search.fit(X_train, y_train)
    best_knn = grid_search.best_estimator_
    y_pred_knn = best_knn.predict(X_test)
    accuracy_knn = accuracy_score(y_test, y_pred_knn)
    st.write(f"Accuracy: {accuracy_knn:.2f}")
    st.write("Best Parameters:", grid_search.best_params_)
    st.write("Best Score:", grid_search.best_score_)
    st.write("Confusion Matrix:")
    st.write(confusion_matrix(y_test, y_pred_knn))
    # st.write("Classification Report:")
    # st.write(classification_report(y_test, y_pred_knn))

    # Decision Tree
    st.write("Decision Tree:")
    dt = DecisionTreeClassifier()
    params = {"max_depth": np.arange(0, 30, 5), "criterion": ["gini", "entropy"]}
    grid_search = GridSearchCV(dt, param_grid=params, cv=5)
    grid_search.fit(X_train, y_train)
    best_dt = grid_search.best_estimator_
    y_pred_dt = best_dt.predict(X_test)
    accuracy_dt = accuracy_score(y_test, y_pred_dt)
    st.write(f"Accuracy: {accuracy_dt:.2f}")
    st.write("Best Parameters:", grid_search.best_params_)
    st.write("Best Score:", grid_search.best_score_)
    st.write("Confusion Matrix:")
    st.write(confusion_matrix(y_test, y_pred_dt))
    # st.write("Classification Report:")
    # st.write(classification_report(y_test, y_pred_dt))

    st.header("6. In the end, the accuracy of Decision Tree > KNN > Logistic Regression")

    #DecisonTree


    return best_log_reg, best_knn




def main():
    st.title("Booking Dataset Analysis using KNN, Decision tree Logistic Regression")
    st.caption("Team Members: \n - Rithik Bal R S, 3122 21 5001 081\n- Roshni Badrinath 3122 21 5001 087")
    st.write("---")

    st.header("1. Import Libraries")
    st.write("Imports necessary libraries like pandas, numpy, matplotlib, seaborn, ydata_profiling, sklearn modules, and warning.")
    st.write("---")

    st.header("2. Load Dataset")
    st.write("> Hotel Booking Cancellation Prediction dataset, a comprehensive collection of data aimed at predicting hotel booking cancellations. This dataset comprises a diverse range of features, including booking details, customer information, and reservation specifics. The information has been meticulously gathered from real-world hotel booking scenarios, ensuring authenticity and relevance for predictive modeling.")
    st.write("The dataset is provided in CSV format")
    st.write("> Booking_ID: Unique identifier for each booking")
    st.write("> number of adults: Number of adults included in the booking")
    st.write("> number of children: Number of children included in the booking")
    st.write("> number of weekend nights: Number of weekend nights included in the booking")
    st.write("> number of week nights: Number of week nights included in the booking")
    st.write("> type of meal: type of meal included in the booking")
    st.write("> car parking space: Indicates whether a car parking space was requested or included in the booking")
    st.write("> room type: Type of room booked")
    st.write("> lead time: Number of days between the booking date and the arrival date")
    st.write("> market segment type: Type of market segment associated with the booking")
    file_path = "/Users/rikky/Downloads/machine/booking.csv"
    booking = load_data(file_path)
    st.write("---")

    # Preprocess Data
    booking = preprocess_data(booking)
    st.write("---")

    # Feature Selection
    X, target = feature_selection(booking)
    st.write("---")

    # Train Models
    train_models(X, target)
    st.write("---")

if __name__ == "__main__":
    main()
