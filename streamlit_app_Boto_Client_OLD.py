import streamlit as st
import pandas as pd
import boto3
import json
from loguru import logger

# Define the rubric for scoring
rubric = {
    1: "The model fails to understand the context of user inputs and provides responses that are irrelevant or inappropriate.",
    2: "The model occasionally understands the context but often provides responses that are incomplete or only partially appropriate.",
    3: "The model generally understands the context and provides appropriate responses, though some responses may be lacking in detail or accuracy.",
    4: "The model consistently understands the context and provides suitable and detailed responses, with only occasional minor inaccuracies or omissions.",
    5: "The model excels in understanding the context and consistently provides highly relevant, detailed, and accurate responses."
}

def get_bedrock_client():
    return boto3.client(service_name='bedrock-runtime', region_name='us-east-1')  # Adjust region as necessary

def query_model(client, model_id, question):
    try:
        formatted_prompt = f"Human: {question}\nAssistant:"
        max_tokens = 4096
        temperature = 0.0
        native_request = {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [
                {"role": "user", "content": formatted_prompt}
            ]
        }
        request = json.dumps(native_request)
        response = client.invoke_model(modelId=model_id, body=request)
        model_response = json.loads(response["body"].read().decode('utf-8'))
        response_text = model_response['choices'][0]['message']['content'] if 'choices' in model_response else model_response['messages'][0]['content']

        return response_text
    except Exception as e:
        logger.error(f"Error invoking model: {str(e)}")
        return f"Error invoking model: {str(e)}"

def evaluate_response(response):
    # Example of a very simple evaluation logic
    if len(response.split()) < 10:
        score = 1
    elif len(response.split()) < 20:
        score = 2
    elif len(response.split()) < 40:
        score = 3
    elif len(response.split()) < 80:
        score = 4
    else:
        score = 5
    return score, rubric[score]

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
            evaluation_results = df['Candidate Answer'].apply(evaluate_response)
            df['Evaluator Rating'] = [result[0] for result in evaluation_results]
            df['Evaluator Rating Justification'] = [result[1] for result in evaluation_results]

            st.write("Generated Answers and Evaluation:")
            st.dataframe(df)

            df.to_csv("streamlit_llm_answers.csv", index=False)
            st.download_button("Download CSV", df.to_csv(index=False), "streamlit_llm_answers.csv", "text/csv", key='download-csv')

            st.success("Generated answers and evaluation saved to streamlit_llm_answers.csv")

if __name__ == "__main__":
    main()
