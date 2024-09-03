import sys
import os
import argparse
import mlflow
from dotenv import load_dotenv
from model_retriever import get_azure_gpt_model, get_parameter_store
from functions import (
    display_audio_player,
    download_from_s3,
    string_to_yaml,
    extract_text_from_pdf,
    extract_text_from_txt,
    ask_question, 
    display_chat,
)
from helper import call_transcription_model_change, initialize_chat_with_file, call_audio_s3_file_upload, file_upload
from constants import (
    TEMP_DOWNLOAD_PATH,
    MODEL_DICT,
    chat_styles,
    GPT_LLM_MODEL_DICT,
    CLAUDE_LLM_MODEL_DICT,
    LLAMA3_LLM_MODEL_DICT,
    LLAMA2_LLM_MODEL_DICT,
    MISTRAL_LLM_MODEL_DICT,
    AUDIO_FILE_TYPES,
)
import streamlit as st
import streamlit_authenticator as stauth
from loguru import logger

from upload_file_gpt import create_message_content, image_to_base64_images

st.set_page_config(layout="wide")
llm_model = None

### setup.py ###
def setup_mlflow(tracking_uri = "http://localhost:5001"):
    """Configures MLflow with the specified tracking server URI."""
    mlflow.set_tracking_uri(tracking_uri)

def create_experiment(experiment_name = "llm_patterns_"):
    """Creates a new experiment in MLflow."""
    existing_experiment = mlflow.get_experiment_by_name(experiment_name)
    if existing_experiment:
        return existing_experiment.experiment_id
    else:
        return mlflow.create_experiment(experiment_name)
    
def load_css(file_path):
    with open(file_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

MLFLOW_TRACKING_URI                     = "https://mlflow.dev.fhmc.xpanse-ai.com"
MLFLOW_EXPERIMENT_NAME                  = "chat-with-file"

def initialize_session_state():
    logger.info("Setting up mlflow...")
    os.environ['MLFLOW_TRACKING_USERNAME'] = 'admin'
    os.environ['MLFLOW_TRACKING_PASSWORD'] = 'password'
    setup_mlflow(MLFLOW_TRACKING_URI)
    mlflow.create_experiment(MLFLOW_EXPERIMENT_NAME)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    if "authentication_status" not in st.session_state:
        st.session_state["authentication_status"] = None
    if "clicked" not in st.session_state:
        st.session_state.clicked = False
    if os.path.exists(TEMP_DOWNLOAD_PATH):
        os.remove(TEMP_DOWNLOAD_PATH)
    initialize_chat_with_file()


def authenticate(dev_mode):
    if dev_mode:
        st.session_state["authentication_status"] = True
        st.session_state["name"] = "Developer"
        return

    config = get_parameter_store("/podium-ia-call-summerization-poc/auth.yml")
    config = string_to_yaml(config)
    authenticator = stauth.Authenticate(
        config["credentials"],
        config["cookie"]["name"],
        config["cookie"]["key"],
        config["cookie"]["expiry_days"],
    )
    name, authentication_status, username = authenticator.login()
    st.session_state["authentication_status"] = authentication_status
    st.session_state["name"] = name
    return authenticator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", action="store_true", help="Run in development mode")
    args = parser.parse_args()

    sys.path.append(os.path.dirname(os.path.realpath(__file__)))
    load_dotenv("../.env")
    load_css("styles.css")
    if "instialized" not in st.session_state:
        st.session_state["instialized"] = False 
        initialize_session_state()
    authenticator = authenticate(args.dev)

    if st.session_state["authentication_status"]:
        display_authenticated_content(args.dev, authenticator)
    elif st.session_state["authentication_status"] is False:
        st.error("Username/password is incorrect")
    else:
        st.warning("Please enter your username and password")


def display_authenticated_content(dev_mode, authenticator):
    with st.sidebar:
        st.write(f'Welcome *{st.session_state["name"]}*')
        if not dev_mode:
            authenticator.logout()
        else:
            st.warning("The app is running in Dev mode, auth is bypassed")

    transcription_model = st.sidebar.selectbox(
        "Which model would you like to use for transcription?",
        ("whisper", "amazon_transcribe"), on_change=call_transcription_model_change, key="transcription_model_select")
    
    llm_model = st.sidebar.selectbox(
        "Which model would you like to use for chat with file feature?",
        (*list(GPT_LLM_MODEL_DICT.keys()), *list(CLAUDE_LLM_MODEL_DICT.keys()), *list(LLAMA2_LLM_MODEL_DICT.keys()), *list(LLAMA3_LLM_MODEL_DICT.keys()), *list(MISTRAL_LLM_MODEL_DICT.keys())))

    logger.info(f"Selected transcription model: {transcription_model}")
    logger.info(f"Selected model: {llm_model}")

    st.logo("./images/xpanse_logo_2clr.webp")
    st.markdown(
        '<h1 class="title">Chat with file usecase powered by Halo</h1>',
        unsafe_allow_html=True,
    )
    st.warning("This app is intended for demonstration purposes only.")

    file_upload_container = st.container(border=True)
    audio_player_container = st.container(border=False)
    s3_uri = None
    file = None
    file = file_upload_container.file_uploader(
        "Upload the file", type=["pdf", "txt", "png", "jpeg", "wav", "mp3"], accept_multiple_files=False, on_change=file_upload, key="file"
    )
    file_upload_container.markdown(
            "<h5 style='text-align: center;'> OR </h5>", unsafe_allow_html=True
        )
    file_upload_container.text_input("Enter the S3 URI of the file", on_change=call_audio_s3_file_upload, key="s3_uri")

    if file and (file.type in AUDIO_FILE_TYPES):
        display_audio_player(file, audio_player_container)
        
    s3_uri = st.session_state.s3_uri
    if s3_uri: 
        download_from_s3(s3_uri, TEMP_DOWNLOAD_PATH)       
        display_audio_player(TEMP_DOWNLOAD_PATH, audio_player_container)

    if file or s3_uri:
        if st.session_state["raw_transcription"] is not None:
            with st.expander("Raw Transcription"):
                transcription = st.session_state["raw_transcription"]   
                line_count = transcription.count('\n') + 1
                text_height = max(100, min(line_count * 20, 600))
                st.text_area("", value=st.session_state["raw_transcription"], height=text_height)
        # logger.debug("question: ", st.session_state["question"])
        st.text_input("Ask a question about the file content.", value=st.session_state["question"], key="new_question")
        st.session_state["question"] = st.session_state["new_question"]
        
        
        if st.button("Get Answer"):
            # if st.session_state["question"]:
            for question in open("questions.json")["questions"]:
                st.session_state["question"] = question
                datatype = None
                if file:
                    if file.type == "application/pdf":
                        text = extract_text_from_pdf(file)

                    elif file.type == "text/plain":
                        text = extract_text_from_txt(file)

                    elif file.type == "image/png" or file.type == "image/jpeg":
                        images = image_to_base64_images(file)
                        # content = create_message_content(images)
                        datatype = "image"

                    elif file.type in AUDIO_FILE_TYPES:
                        text = st.session_state["raw_transcription"]
                elif s3_uri:
                    text = st.session_state["raw_transcription"]
                with st.spinner("Generating response..."):
                    if datatype == "image":
                        answer, time_taken = ask_question(images, st.session_state["question"], MODEL_DICT[llm_model], datatype)
                    else:
                        answer, time_taken = ask_question(text, st.session_state["question"], MODEL_DICT[llm_model])
                # with open("test_summary_sonnet.txt", "a") as f:
                #     f.write(f"Question: {st.session_state['question']}\nAnswer: {answer}\nTime taken: {time_taken}\n\n")
                # # send to mlflow
                
                with mlflow.start_run(run_name=f"Chat with document with {llm_model}"):
                    mlflow.log_param("Question", st.session_state['question'])
                    mlflow.log_param("Answer", answer)
                    mlflow.log_metric("Time Taken (s)", time_taken)

                
                st.session_state['messages'].append({"role": "logs", "time_taken": time_taken})
                st.session_state['messages'].append({"role": "Assistant", "content": answer.replace("$", "\$")})
                st.session_state['messages'].append({"role": "User", "content": st.session_state["question"]})
                st.markdown(chat_styles, unsafe_allow_html=True)
                display_chat()                
            else:
                st.write("Please ask a question.")
        

if __name__ == "__main__":
    main()