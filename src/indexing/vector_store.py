import os
import time
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

class JobVectorStore:
    """공고 데이터 임베딩 및 벡터 저장소 관리 클래스"""

    def __init__(self, vector_db_path: str = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "vector_db", "faiss_index")):
        self.vector_db_path = vector_db_path
        self.embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")
        self.vector_store = None

    def create_vector_store(self, documents: List[Document], batch_size: int = 50):
        """대용량 문서를 배치 단위로 나누어 벡터 스토어를 생성하고 저장합니다."""
        if not documents:
            print("❌ 저장할 문서가 없습니다.")
            return

        print(f"🚀 총 {len(documents)}개의 문서를 {batch_size}개씩 나누어 처리 중... (로컬 임베딩)")

        # 첫 번째 배치로 초기 벡터 스토어 생성
        first_batch = documents[:batch_size]
        self.vector_store = FAISS.from_documents(first_batch, self.embeddings)
        total_batches = (len(documents) + batch_size - 1) // batch_size
        print(f"📦 완료: 1/{total_batches} 배치 ({len(first_batch)}개)")

        # 나머지 배치들 추가
        batch_num = 2
        for i in range(batch_size, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            self.vector_store.add_documents(batch)
            print(f"📦 완료: {batch_num}/{total_batches} 배치 ({len(batch)}개)")
            batch_num += 1
        
        # 벡터 스토어 디렉토리 생성 및 저장
        os.makedirs(os.path.dirname(self.vector_db_path), exist_ok=True)
        self.vector_store.save_local(self.vector_db_path)
        print(f"✅ 벡터 DB가 '{self.vector_db_path}'에 안전하게 저장되었습니다.")

    def load_vector_store(self):
        """로컬에 저장된 벡터 스토어를 불러옵니다."""
        if os.path.exists(self.vector_db_path):
            # 보안 참고: allow_dangerous_deserialization=True는 로컬 환경용 옵션
            self.vector_store = FAISS.load_local(
                self.vector_db_path, 
                self.embeddings, 
                allow_dangerous_deserialization=True
            )
            return self.vector_store
        else:
            return None

    def get_retriever(self, search_kwargs={"k": 5}):
        """저장된 벡터 DB로부터 retriever 객체를 반환합니다."""
        if not self.vector_store:
            self.load_vector_store()
            
        if self.vector_store:
            return self.vector_store.as_retriever(search_kwargs=search_kwargs)
        return None

if __name__ == "__main__":
    import json
    import time
    from src.preprocessing.preprocess import JobPreprocessor
    
    # 절대 경로로 확실하게 지정
    sample_file_path = r"C:\Users\kobin\AppData\Local\Google\Cloud SDK\project1_0220\data\raw\real_saramin_data.json"
    try:
        if not os.path.exists(sample_file_path):
            print(f"❌ '{sample_file_path}' 파일이 없습니다.")
        else:
            with open(sample_file_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
                
            print(f"📄 총 {len(raw_data)}건의 데이터 로드 완료.")
            
            # 3. 벡터 DB 생성 및 저장
            vector_store = JobVectorStore()
            vector_store.create_vector_store(docs, batch_size=100)
            
            print(f"✅ 테스트 성공: {len(docs)}개의 대용량 데이터가 저장되었습니다.")
            
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
