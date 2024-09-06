import streamlit as st
import pandas as pd
from langchain.llms.bedrock import Bedrock
from langchain_community.chat_models import BedrockChat

def get_bedrock_client(model_id):
    print(f"Using model ID: {model_id}")  # Debugging line to confirm model ID
    if "claude" in model_id.lower():
        return BedrockChat(model_id=model_id)
    else:
        return Bedrock(model_id=model_id)

def query_model(bedrock_client, question):
    try:
        messages = [{"role": "user", "content": question}]
        response = bedrock_client.generate(messages=messages, max_tokens=500)
        return response.text  # Adjusted assuming response.text is the correct way to access the response
    except Exception as e:
        return f"Error invoking model: {str(e)}"

def main():
    st.title("AWS Bedrock LLM Evaluator")

    uploaded_file = st.file_uploader("Upload CSV with Questions column", type="csv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Uploaded CSV:")
        st.dataframe(df)

        model_id_mapping = {
            "Claude 3.5 Sonnet": "anthropic.claude-3-5-sonnet-20240620-v1:0",  # Example ID, verify correctness
            "Mistral Large 2 (24.07)": "mistral.mistral-large-2407-v1:0",  # Example ID, verify correctness
            "Mistral Large": "mistral.mistral-large-2402-v1:0",  # Example ID, verify correctness
        }

        candidate_model = st.selectbox("Select Candidate Model", list(model_id_mapping.keys()), index=0)
        evaluator_model = st.selectbox("Select Evaluator Model", list(model_id_mapping.keys()), index=1)

        if st.button("Generate Answers and Evaluate"):
            candidate_bedrock_client = get_bedrock_client(model_id_mapping[candidate_model])
            evaluator_bedrock_client = get_bedrock_client(model_id_mapping[evaluator_model])

            df['Candidate Answer'] = df['Questions'].apply(lambda x: query_model(candidate_bedrock_client, x))
            df['Evaluator Rating'] = df.apply(lambda row: query_model(evaluator_bedrock_client, f"Evaluate the following answer on a scale of 1 to 5: {row['Candidate Answer']}"), axis=1)
            df['Evaluator Rating Justification'] = df.apply(lambda row: query_model(evaluator_bedrock_client, f"Provide justification for the rating of {row['Evaluator Rating']}"), axis=1)

            st.write("Generated Answers and Evaluation:")
            st.dataframe(df)

            df.to_csv("streamlit_llm_answers.csv", index=False)
            st.download_button("Download CSV", df.to_csv(index=False), "streamlit_llm_answers.csv", "text/csv", key='download-csv')

            st.success("Generated answers and evaluation saved to streamlit_llm_answers.csv")

if __name__ == "__main__":
    main()
