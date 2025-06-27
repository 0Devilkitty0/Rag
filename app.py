import os
import streamlit as st
import tempfile

# 기존 Chroma 관련 임포트 변경
# from langchain.vectorstores import Chroma # 제거
from langchain_community.vectorstores import FAISS # FAISS 임포트
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories.streamlit import StreamlitChatMessageHistory

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
# from langchain.core.output_parsers import StrOutputParser

# pysqlite3 관련 코드 제거
# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# langchain_chroma 임포트 제거
# from langchain_chroma import Chroma 

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

#cache_resource로 한번 실행한 결과 캐싱해두기
@st.cache_resource
def load_and_split_pdf(file_path):
    loader = PyPDFLoader(file_path)
    return loader.load_and_split()

# 텍스트 청크들을 FAISS 안에 임베딩 벡터로 저장 (Chroma 대신 FAISS 사용)
@st.cache_resource
def create_vector_store(_docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    split_docs = text_splitter.split_documents(_docs)
    # persist_directory는 FAISS에서는 일반적으로 사용되지 않습니다.
    # FAISS는 인메모리(in-memory)로 동작하며, 저장하려면 FAISS.save_local()을 사용해야 합니다.
    # 여기서는 cache_resource 덕분에 메모리에 유지되므로 별도 저장은 필요 없습니다.
    vectorstore = FAISS.from_documents(
        split_docs, 
        OpenAIEmbeddings(model='text-embedding-3-small')
    )
    return vectorstore

# FAISS는 별도 로드 로직 필요 없음 (cache_resource에 의해 관리)
@st.cache_resource
def get_vectorstore(_docs):
    # FAISS는 persist_directory 개념을 직접 사용하지 않으므로,
    # 항상 create_vector_store(_docs)를 호출하도록 변경합니다.
    # @st.cache_resource 덕분에 실제로는 한 번만 실행됩니다.
    return create_vector_store(_docs)
    
# PDF 문서 로드-벡터 DB 저장-검색기-히스토리 모두 합친 Chain 구축
@st.cache_resource
def initialize_components(selected_model):
    file_path = "data/대한민국헌법(헌법)(제00010호)(19880225).pdf"
    
    if not os.path.exists(file_path):
        st.error(f"Error: PDF file not found at {file_path}. Please ensure it's in the 'data' folder in your GitHub repository.")
        st.stop() 

    pages = load_and_split_pdf(file_path)
    vectorstore = get_vectorstore(pages)
    retriever = vectorstore.as_retriever()

    # 채팅 히스토리 요약 시스템 프롬프트
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("history"),
            ("human", "{input}"),
        ]
    )

    # 질문-답변 시스템 프롬프트
    qa_system_prompt = """You are an assistant for question-answering tasks. \
    Use the following pieces of retrieved context to answer the question. \
    If you don't know the answer, just say that you don't know. \
    Keep the answer perfect. please use imogi with the answer.
    대답은 한국어로 하고, 존댓말을 써줘.\

    {context}"""
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("history"),
            ("human", "{input}"),
        ]
    )

    llm = ChatOpenAI(model=selected_model)
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    Youtube_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, Youtube_chain)
    return rag_chain

# Streamlit UI
st.header("헌법 Q&A 챗봇 💬 📚")
option = st.selectbox("Select GPT Model", ("gpt-4o-mini", "gpt-3.5-turbo-0125"))
rag_chain = initialize_components(option)
chat_history = StreamlitChatMessageHistory(key="chat_messages")

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    lambda session_id: chat_history,
    input_messages_key="input",
    history_messages_key="history",
    output_messages_key="answer",
)


if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", 
                                     "content": "헌법에 대해 무엇이든 물어보세요! ⚖️"}]

for msg in chat_history.messages:
    st.chat_message(msg.type).write(msg.content)


if prompt_message := st.chat_input("궁금한 점을 입력해주세요..."):
    st.chat_message("human").write(prompt_message)
    with st.chat_message("ai"):
        with st.spinner("생각 중입니다... 🧐"):
            config = {"configurable": {"session_id": "any"}}
            response = conversational_rag_chain.invoke(
                {"input": prompt_message},
                config)
            
            answer = response['answer']
            st.write(answer)
            with st.expander("참고 문서 확인 📄"):
                if 'context' in response and response['context']:
                    for i, doc in enumerate(response['context']):
                        st.markdown(f"**문서 {i+1}:**")
                        source_info = doc.metadata.get('source', '출처 알 수 없음')
                        page_info = doc.metadata.get('page', '페이지 정보 없음')
                        st.markdown(f"**출처:** {source_info}, **페이지:** {page_info}")
                        st.write(doc.page_content)
                        st.markdown("---")
                else:
                    st.info("관련 문서를 찾을 수 없습니다.")
