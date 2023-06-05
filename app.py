import streamlit as st
from llama_index import download_loader
from pathlib import Path
from llama_index import LLMPredictor, GPTVectorStoreIndex, PromptHelper, ServiceContext
from llama_index import QuestionAnswerPrompt, GPTVectorStoreIndex, SimpleDirectoryReader
from langchain import OpenAI
import os

os.environ["OPENAI_API_KEY"] = st.secrets.openai_api_key

INTRO = "この文章を３０字程度で要約して下さい。　回答後は、必ず'改行'して「ご質問をどうぞ。」を付けて下さい。"

if "qa" not in st.session_state:
    st.session_state.qa = {"pdf": "", "history": []}
#     st.session_state["qa"] = {"pdf": "", "history": [{"role": "Q", "msg": INTRO}]}

# Prompt
QA_PROMPT_TMPL = (
    "下記の情報が与えられています。 \n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "この情報を参照して次の質問に日本語で答えてください: {query_str}\n"
)
QA_PROMPT = QuestionAnswerPrompt(QA_PROMPT_TMPL)

# Class and Function
# class StreamHandler(BaseCallbackHandler):
#     def __init__(self, container, initial_text = ""):
#         self.container = container
#         self.text=initial_text
#     def on_llm_new_token(self, token: str, **kwargs) -> None:
#         self.text+=token 
#         self.container.success(self.text) 

@st.cache_resource
def get_vector_db(uploaded_file):
    with NamedTemporaryFile(dir='.', suffix='.pdf') as f:
        f.write(uploaded_file.getbuffer())
        PDFReader = download_loader("PDFReader")
        loader = PDFReader()
        documents = loader.load_data(file=Path(f.name))
    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="gpt-3.5-turbo", streaming=True))
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)
    index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)
    return index

def store_del_msg():
    st.session_state.qa["history"].append({"role": "Q", "msg": st.session_state.user_input}) # store
    st.session_state.user_input = ""  # del

# View (User Interface)
## Sidebar
st.sidebar.title("ＰＤＦアシスタント")
uploaded_file = st.sidebar.file_uploader("PDFファイルをアップロードして下さい", type=["pdf"])
if uploaded_file is not None:
    if uploaded_file.name != st.session_state.qa.get("pdf", ""):
        st.session_state.qa = {"pdf": uploaded_file.name, "history": []}
    user_input = st.sidebar.text_input("ご質問をどうぞ", key="user_input", on_change=store_del_msg)
    ## Main Content
    if st.session_state.qa["history"]:
        for message in st.session_state.qa["history"]:
#         for message in st.session_state["qa"][1:]:
            if message["role"] == "Q": # Q: Question (User)
                st.info(message["msg"])
            elif message["role"] == "A": # A: Answer (AI Assistant)
                st.success(message["msg"])
            elif message["role"] == "E": # E: Error
                st.error(message["msg"])
    chat_box = st.empty() # Streaming message

    # Model (Business Logic)
    index = get_vector_db(uploaded_file)
    stream_handler = StreamHandler(chat_box)
    engine = index.as_query_engine(text_qa_template=QA_PROMPT)
    if st.session_state.qa["history"]:
        query = st.session_state.qa["history"][-1]["msg"]
        try:
            response = engine.query(query) # Query to ChatGPT
            st.session_state.qa["history"].append({"role": "A", "msg": response})
        except Exception:
            response = "エラーが発生しました！　もう一度、質問して下さい。"
            st.error(response)
            st.session_state.qa.qa["history"].append({"role": "E", "msg": response})
