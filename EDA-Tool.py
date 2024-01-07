# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import warnings
import numpy as np

warnings.filterwarnings("ignore")


# Manual label encoding function
def label_encode(column):
    unique_labels = set(column)
    label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
    encoded_column = [label_mapping[label] for label in column]
    return encoded_column, label_mapping


def main():
    st.title("CSV Data Analytics Tool")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        # Process CSV file using pandas without specifying data types
        df = pd.read_csv(uploaded_file)

        # Display the uploaded data
        st.write("Uploaded Data:", df)

        # Check for null values
        st.subheader("Check for Null Values:")
        null_values = df.isnull().sum()
        st.write("Null Values in Each Column:", null_values)

        # Handling null values
        if null_values.any():
            st.subheader("Handling Null Values:")

            # Option to remove or replace null values
            action = st.radio("Choose action for null values:", ["Remove", "Replace"])

            if action == "Remove":
                # Remove rows with null values
                df = df.dropna()
                st.write("Rows with null values removed.")
            elif action == "Replace":
                # Provide options for replacing null values
                replacement_method = st.selectbox("Select replacement method:", ["Mean", "Median", "Mode", "Custom"])

                if replacement_method == "Mean":
                    # Apply fillna only to numeric columns
                    df_numeric = df.select_dtypes(include=['number'])
                    df[df_numeric.columns] = df[df_numeric.columns].fillna(df_numeric.mean())
                    st.write("Null values replaced with mean.")
                elif replacement_method == "Median":
                    df_numeric = df.select_dtypes(include=['number'])
                    df[df_numeric.columns] = df[df_numeric.columns].fillna(df_numeric.median())
                    st.write("Null values replaced with median.")
                elif replacement_method == "Mode":
                    df_numeric = df.select_dtypes(include=['number'])
                    df[df_numeric.columns] = df[df_numeric.columns].fillna(df_numeric.mode().iloc[0])
                    st.write("Null values replaced with mode.")
                elif replacement_method == "Custom":
                    # Allow users to input custom values for replacement
                    custom_value = st.text_input("Enter custom value for replacement:")
                    df = df.fillna(custom_value)
                    st.write(f"Null values replaced with custom value: {custom_value}")

            # Display the updated DataFrame
            st.write("Updated Data:", df)

            # Option to download the updated CSV
            if st.button("Download Updated CSV"):
                csv_buffer = BytesIO()
                df.to_csv(csv_buffer, index=False)
                csv_buffer.seek(0)
                st.download_button(
                    label="Download CSV",
                    data=csv_buffer,
                    file_name="updated_data.csv",
                    key="download_button"
                )

        # Feature selection
        st.subheader("Feature Selection:")
        target_variable = st.selectbox("Select the target variable", df.columns)

        # Label encoding option for the target variable
        label_encoding = st.checkbox("Apply Label Encoding for Target Variable")
        if label_encoding:
            # Manual label encoding for the target variable
            encoded_column, label_mapping = label_encode(df[target_variable])
            df[target_variable + '_encoded'] = encoded_column

            # Displaying the label mapping
            st.write(f"Label Mapping for {target_variable}: {label_mapping}")

            # Replace the original target variable with the encoded one
            df = df.drop(columns=[target_variable])  # Drop the original target variable column
            df[target_variable] = encoded_column  # Add the encoded column as the new target variable

        selected_features = st.multiselect("Select features to visualize with the target", df.columns)

        # Check for non-numeric columns in selected features
        numeric_selected_features = [col for col in selected_features if df[col].dtype in ['int64', 'float64']]

        # Additional plot options
        st.subheader("Additional Plots:")
        plot_options = ["Histogram", "Box Plot", "Pair Plot", "Scatter Plot", "Correlation Matrix"]
        selected_plots = st.multiselect("Select additional plot types:", plot_options)

        # Generate plots based on user selection
        for selected_plot in selected_plots:
            if selected_plot == "Histogram":
                for feature in numeric_selected_features:
                    st.subheader(f"Histogram for {feature}")
                    fig, ax = plt.subplots()
                    ax.hist(df[feature], bins=20, color='skyblue', edgecolor='black')
                    st.pyplot(fig)
                    plt.close(fig)
                st.markdown("A histogram provides a visual representation of the distribution of a numeric feature. "
                            "It shows the frequency of values within certain ranges.")

            elif selected_plot == "Box Plot":
                for feature in numeric_selected_features:
                    st.subheader(f"Box Plot for {feature}")
                    fig, ax = plt.subplots()
                    sns.boxplot(x=df[feature])
                    st.pyplot(fig)
                    plt.close(fig)
                st.markdown("A box plot displays the distribution and spread of a numeric feature. "
                            "It includes information about the median, quartiles, and potential outliers.")

            elif selected_plot == "Pair Plot":
                st.subheader("Pair Plot for Selected Features")
                # Create a pair plot
                pair_plot = sns.pairplot(df[numeric_selected_features])

                # Save the pair plot to a BytesIO buffer
                buffer_pair_plot = BytesIO()
                pair_plot.savefig(buffer_pair_plot, format='png')
                buffer_pair_plot.seek(0)

                # Display the pair plot
                st.image(buffer_pair_plot, caption="Pair Plot", use_column_width=True, format='png')
                plt.close(pair_plot.fig)
                st.markdown("A pair plot provides scatter plots for all pairs of selected numeric features. "
                            "It helps visualize relationships and identify patterns.")

            elif selected_plot == "Scatter Plot":
                st.subheader("Scatter Plot for Selected Features")
                # Create scatter plots for selected features and target variable
                fig, ax = plt.subplots()
                for feature in numeric_selected_features:
                    ax.scatter(df[feature], df[target_variable], label=feature)
                ax.set_xlabel('Features')
                ax.set_ylabel(target_variable)
                ax.legend()
                st.pyplot(fig)
                plt.close(fig)
                st.markdown("A scatter plot visualizes the relationship between selected numeric features and the target variable. "
                            "It helps identify trends, patterns, and potential correlations.")

            elif selected_plot == "Correlation Matrix":
                st.subheader("Correlation Matrix:")
                correlation_matrix = df[numeric_selected_features + [target_variable]].corr()

                # Display correlation matrix as a heatmap
                sns.set(style="white")
                mask = np.zeros_like(correlation_matrix, dtype=bool)
                mask[np.triu_indices_from(mask)] = True
                fig, ax = plt.subplots(figsize=(10, 8))
                cmap = sns.diverging_palette(220, 20, as_cmap=True)
                sns.heatmap(correlation_matrix, annot=True, cmap=cmap, mask=mask, vmax=.3, center=0,
                            square=True, linewidths=.5, cbar_kws={"shrink": .5})
                st.pyplot(fig)
                plt.close(fig)

                # Calculate feature importance based on correlation with the target variable
                feature_importance = correlation_matrix[target_variable].abs().sort_values(ascending=False)
                most_important_feature = feature_importance.index[1]  # Exclude the target variable
                least_important_feature = feature_importance.index[-1]  # Exclude the target variable

                # Display most and least important features
                st.subheader("Feature Importance:")
                st.write(f"Most Important Feature: {most_important_feature}")
                st.write(f"Least Important Feature: {least_important_feature}")

if __name__ == "__main__":
    main()
