import streamlit as st
import boto3
import pandas as pd
import json

def query_model(client, model_info, question):
    try:
        model_id = model_info["model_id"]
        content_type = model_info["content_type"]

        if content_type == "application/json":
            input_data = json.dumps({
                "prompt": question,
                "max_tokens_to_sample": 500  # Adjust this value as necessary
            })
        else:
            input_data = question

        response = client.invoke_model(
            modelId=model_id,
            contentType=content_type,
            body=input_data,
        )
        
        return response['body'].read().decode('utf-8')
    except Exception as e:
        return f"Error invoking model: {str(e)}"  # Capture and return errors

def main():
    st.title("AWS Bedrock LLM Evaluator")

    uploaded_file = st.file_uploader("Upload CSV with Questions column", type="csv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Uploaded CSV:")
        st.dataframe(df)

        model_id_mapping = {
            #"Anthropic Claude 1": {"model_id": "anthropic.claude-v1", "content_type": "text/plain"},
            #"Anthropic Claude 2": {"model_id": "anthropic.claude-v2", "content_type": "application/json"},
            #"Anthropic Claude 3.0": {"model_id": "anthropic.claude-v3.0", "content_type": "application/json"},
            "Claude 3.5 Sonnet": {"model_id": "anthropic.claude-3-5-sonnet-20240620-v1:0", "content_type": "application/json"},
            "Mistral Large 2 (24.07)": {"model_id": "mistral.mistral-large-2407-v1:0", "content_type": "application/json"},
            "Mistral Large": {"model_id": "mistral.mistral-large-2402-v1:0", "content_type": "application/json"},
        }

        candidate_model = st.selectbox("Select Candidate Model", list(model_id_mapping.keys()), index=list(model_id_mapping.keys()).index("Mistral Large 2 (24.07)"))
        evaluator_model = st.selectbox("Select Evaluator Model", list(model_id_mapping.keys()), index=list(model_id_mapping.keys()).index("Claude 3.5 Sonnet"))

        if st.button("Generate Answers and Evaluate"):
            client = boto3.client('bedrock-runtime', region_name='us-east-1')  # Ensure the correct AWS region

            df['Candidate Answer'] = df['Questions'].apply(lambda x: query_model(client, model_id_mapping[candidate_model], x))
            df['Evaluator Rating'] = df.apply(lambda row: query_model(client, model_id_mapping[evaluator_model], f"Evaluate the following answer on a scale of 1 to 5: {row['Candidate Answer']}"), axis=1)
            df['Evaluator Rating Justification'] = df.apply(lambda row: query_model(client, model_id_mapping[evaluator_model], f"Provide justification for the rating of {row['Evaluator Rating']}"), axis=1)

            st.write("Generated Answers and Evaluation:")
            st.dataframe(df)

            df.to_csv("streamlit_llm_answers.csv", index=False)
            st.download_button("Download CSV", df.to_csv(index=False), "streamlit_llm_answers.csv", "text/csv", key='download-csv')

            st.success("Generated answers and evaluation saved to streamlit_llm_answers.csv")

if __name__ == "__main__":
    main()
