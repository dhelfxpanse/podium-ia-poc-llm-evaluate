
import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO

# Mock function to simulate evaluatee model output
def generate_evaluatee_answer(question, citations):
    return f"Generated answer for: {question}"

# Mock function to simulate evaluator model rating and justification
def evaluate_answer(answer, citations):
    rating = np.random.randint(1, 6)  # Random rating between 1 and 5
    justification = f"Justification for rating {rating}: The answer was evaluated based on the provided citations."
    return rating, justification

# Set up the Streamlit app
st.title("Xpanse Podium IA LLM Evaluate App")

# File uploader for input CSV
uploaded_file = st.file_uploader("Upload a CSV file with questions and citations", type="csv")

if uploaded_file is not None:
    # Read the uploaded CSV file
    df = pd.read_csv(uploaded_file)
    
    # Display the uploaded data
    st.write("Uploaded Data:")
    st.dataframe(df)
    
    # Ensure the CSV has the correct columns
    if len(df.columns) == 2:
        # Initialize lists to hold results
        answers = []
        ratings = []
        justifications = []
        
        # Loop through the rows to evaluate each question
        for index, row in df.iterrows():
            question, citations = row[0], row[1]
            answer = generate_evaluatee_answer(question, citations)
            rating, justification = evaluate_answer(answer, citations)
            answers.append(answer)
            ratings.append(rating)
            justifications.append(justification)
        
        # Create a results DataFrame
        results_df = pd.DataFrame({
            "Question": df.iloc[:, 0],
            "Citations": df.iloc[:, 1],
            "Evaluatee Answer": answers,
            "Evaluator Rating": ratings,
            "Evaluator Rating Justification": justifications
        })
        
        # Display the results
        st.write("Evaluation Results:")
        st.dataframe(results_df)
        
        # Option to save the results to a CSV file
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="Download Results as CSV",
            data=csv,
            file_name="evaluation_results.csv",
            mime="text/csv"
        )
        
        # Display a statistical summary of the ratings
        st.write("Statistical Summary of Ratings:")
        st.write(results_df["Evaluator Rating"].describe())
    else:
        st.error("Uploaded CSV does not have the required format. It should have exactly 2 columns.")
else:
    st.info("Please upload a CSV file to proceed.")

# Sidebar for model selection
st.sidebar.header("Model Selection")
supported_models = (
    "GPT-4o",
    "mistral.mistral-large-2402-v1:0",
    "anthropic.claude-3-5-sonnet-20240620-v1:0",
)

evaluatee_model = st.sidebar.selectbox("Select Evaluatee Model", supported_models)
evaluator_model = st.sidebar.selectbox("Select Evaluator Model", supported_models)
