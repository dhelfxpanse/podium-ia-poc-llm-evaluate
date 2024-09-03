import streamlit as st
import pandas as pd
import boto3
import json
from loguru import logger
 
def get_bedrock_client():
    return boto3.client(service_name='bedrock-runtime', region_name='us-east-1')  # Adjust region as necessary
 
def query_model(client, model_id, question):
    try:
        if model_id == "mistral.mistral-large-2407-v1:0":
            formatted_prompt = f"Human: {question}\nAssistant:"
            client = boto3.client('bedrock-runtime', 'us-west-2')
            max_tokens = 4096
            temperature = 0.
            native_request = {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [
                {"role": "user", "content": formatted_prompt}
            ]}
        # Format the prompt according to the model's requirements
        else:
            formatted_prompt = f"Human: {question}\nAssistant:"
            max_tokens = 4096
            temperature = 0.
            native_request = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [
                {"role": "user", "content": formatted_prompt}
            ]
        }   
        request = json.dumps(native_request)
        response = client.invoke_model(modelId=model_id, body=request)
        model_response = json.loads(response["body"].read().decode('utf-8'))
        if model_id == "mistral.mistral-large-2407-v1:0":
            response_text = model_response['choices'][0]['message']['content']
        else:
            response_text = model_response['content'][0]['text']
        
        return response_text
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
            "Claude 3.5 Sonnet": "anthropic.claude-3-5-sonnet-20240620-v1:0",
            "Mistral Large 2 (24.07)": "mistral.mistral-large-2407-v1:0",
        }
 
        candidate_model = st.selectbox("Select Candidate Model", list(model_id_mapping.keys()), index=0)
        evaluator_model = st.selectbox("Select Evaluator Model", list(model_id_mapping.keys()), index=1)
 
        if st.button("Generate Answers and Evaluate"):
            bedrock_client = get_bedrock_client()
 
            df['Candidate Answer'] = df['Questions'].apply(lambda x: query_model(bedrock_client, model_id_mapping[candidate_model], x))
            df['Evaluator Rating'] = df.apply(lambda row: query_model(bedrock_client, model_id_mapping[evaluator_model], f"Evaluate the following answer on a scale of 1 to 5: {row['Candidate Answer']}"), axis=1)
            df['Evaluator Rating Justification'] = df.apply(lambda row: query_model(bedrock_client, model_id_mapping[evaluator_model], f"Provide justification for the rating of {row['Evaluator Rating']}"), axis=1)
 
            st.write("Generated Answers and Evaluation:")
            st.dataframe(df)
 
            df.to_csv("streamlit_llm_answers.csv", index=False)
            st.download_button("Download CSV", df.to_csv(index=False), "streamlit_llm_answers.csv", "text/csv", key='download-csv')
 
            st.success("Generated answers and evaluation saved to streamlit_llm_answers.csv")
 
if __name__ == "__main__":
    main()