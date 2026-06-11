import os
import tempfile

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.callbacks import BaseCallbackHandler

st.title("📄 PDF File Reader")
st.write("----------------")

openai_key = st.text_input("OPENAI_API_KEY", type="password")

uploaded_file = st.file_uploader("PDF 파일을 올려주세요", type=["pdf"])
st.write("----------------")

def pdf_to_document(uploaded_file):
    """Streamlit 업로드 PDF를 LangChain Document 형태로 변환"""
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)

    with open(temp_filepath, "wb") as f:
        f.write(uploaded_file.getvalue())

    loader = PyPDFLoader(temp_filepath)
    pages = loader.load()
    return pages

class StreamHandler(BaseCallbackHandler):
    """GPT가 토큰을 생성할 때마다 Streamlit 화면에 실시간으로 출력하는 Handler"""
    def __init__(self, container):
        self.container = container
        self.text = ""

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        # 새 토큰 누적
        self.text += token
        # 화면 갱신
        self.container.markdown(self.text)

if uploaded_file is not None:
    pages = pdf_to_document(uploaded_file)
    st.success(f"PDF 페이지 : {len(pages)}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    texts = text_splitter.split_documents(pages)
    st.info(f"문서 조각 : {len(texts)}")

    embeddings = OpenAIEmbeddings(api_key=openai_key)

    db = Chroma.from_documents(
        documents=texts,
        embedding=embeddings
    )

    retriever = db.as_retriever(
        search_kwargs={"k": 3}
    )

    st.header("PDF에게 질문하세요")
    question = st.text_input("질문 입력")

    if st.button("질문하기"):
        if question == "":
            st.warning("질문을 입력하세요")
        else:
            with st.spinner("답변 생성중..."):
                # 1. 답변을 실시간으로 출력할 빈 공간(Container) 생성
                chat_box = st.empty()
                handler = StreamHandler(chat_box)

                # 2. LLM 설정 (스트리밍 활성화 및 콜백 핸들러 연결)
                llm = ChatOpenAI(
                    model="gpt-4o-mini",  # gpt-4.1-mini의 오타 수정 (gpt-4o-mini 추천)
                    temperature=0,
                    api_key=openai_key,
                    streaming=True,
                    callbacks=[handler]
                )

                # 3. 프롬프트 템플릿 설정
                prompt = ChatPromptTemplate.from_template(
                    """
                    당신은 PDF 분석 AI 입니다.
                    Context: {context}
                    Question: {input}
                    답변:
                    """
                )

                # 4. 문서 결합 체인 및 리트리버 체인 생성
                document_chain = create_stuff_documents_chain(llm, prompt)
                qa_chain = create_retrieval_chain(retriever, document_chain)

                # 5. 체인 실행 (이 과정에서 스트리밍이 수행됩니다)
                response = qa_chain.invoke({"input": question})
                
                # 6. 스트리밍 완료 후, 최종 답변을 화면에 확실하게 고정하여 출력
                chat_box.markdown(response["answer"])