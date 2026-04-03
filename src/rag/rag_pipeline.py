import os
import json
import time
import streamlit as st
from typing import List, Dict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from src.indexing.vector_store import JobVectorStore

try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass

class JobRAGPipeline:
    """분석가 관점의 고도화된 RAG 파이프라인 클래스"""

    def __init__(self, model_name: str = "gemini-flash-latest"):
        self.api_key = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0,
            google_api_key=self.api_key
        )
        self.vector_store = JobVectorStore()
        self.chain = None
        self._initialize_rag_chain()

    def _initialize_rag_chain(self):
        """RAG 체인을 초기화합니다."""
        retriever = self.vector_store.get_retriever(search_kwargs={"k": 3})
        if not retriever:
            return

        system_prompt = (
            "당신은 친절한 커리어 전문가입니다. "
            "제공된 데이터(Context)를 바탕으로 사용자에게 최적의 공고를 추천하세요.\n\n"
            "**[출력 필수 규칙]**\n"
            "1. 모든 답변에서 공고 링크(URL)는 절대 포함하지 마세요.\n"
            "2. 매칭 점수가 80% 이상인 것만 '추천'으로 분류하고, 나머지는 '유사 공고'로 분류하세요.\n"
            "3. 핵심 기술 스택(Skills)을 별도로 강조해주세요.\n"
            "4. 각 공고의 특징과 추천 이유를 친절하게 설명하세요.\n\n"
            "**[데이터(Context)]**\n{context}\n\n"
            "사용자 질문: {question}"
        )

        prompt = ChatPromptTemplate.from_template(system_prompt)

        # 문서 포맷팅 시 URL 제외
        def format_docs_with_scores(docs_with_scores):
            formatted = []
            for i, (doc, score) in enumerate(docs_with_scores):
                confidence = max(0, 100 - (score * 100))
                meta = doc.metadata
                text = f"공고 {i+1} (매칭율: {confidence:.1f}%)"
                text += f"\n- 회사: {meta['company']}\n- 기술: {meta['skills']}\n- 경력: {meta['exp_category']}"
                text += f"\n- 내용: {doc.page_content}"
                formatted.append(text)
            return "\n\n".join(formatted)

        self.chain = (
            {"context": lambda x: format_docs_with_scores(self.vector_store.vector_store.similarity_search_with_score(x, k=3)),
             "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

    def ask(self, user_query: str) -> str:
        """사용자의 질문에 대해 점수 기반의 정제된 답변을 반환합니다."""
        if not self.chain:
            self._initialize_rag_chain()
            if not self.chain:
                return "벡터 DB를 불러올 수 없습니다. 관리자에게 문의하세요."

        try:
            return self.chain.invoke(user_query)
        except Exception as e:
            print(f"[에러 로그] {e}")
            return "현재 답변 생성 중 문제가 발생했습니다. (API 한도 확인 필요)"
