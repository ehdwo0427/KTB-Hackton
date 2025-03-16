import os
import logging

from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader


class SentenceTransformerWrapper:
    """FAISS에서 사용할 수 있도록 `SentenceTransformer`을 래핑한 클래스"""
    
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name, trust_remote_code=True)

    def embed_documents(self, texts: list) -> list:
        """문서 리스트를 벡터로 변환"""
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> list:
        """쿼리를 벡터로 변환"""
        return self.model.encode([text], normalize_embeddings=True).tolist()[0]
    

class RAGRetriever:
    """벡터 스토어에서 유사한 개발 용어를 검색하는 RAG 기능"""
    
    vector_store_path = "./vector_store/faiss"
    pdf_path = "./vector_store/vector_store_pdf.pdf"
    embedding_model = SentenceTransformerWrapper("intfloat/multilingual-e5-large")
    vector_store = None  # 벡터 스토어 저장
    index = None  # 벡터 인덱스 저장

    @classmethod
    def search_similar_terms(cls, query: str, top_k: int = 3):
        """쿼리와 가장 유사한 개발 용어 검색"""
        if cls.vector_store is None:
            logging.info("벡터 스토어 검색중..")
        cls._load_vector_store()
        results = cls.vector_store.similarity_search(query, k=top_k)
        return [res.page_content for res in results]
    
    @classmethod
    def _load_vector_store(cls):
        """벡터 스토어 로드"""
        if os.path.exists(cls.vector_store_path):
            cls.vector_store = FAISS.load_local(cls.vector_store_path, cls.embedding_model.embed_query, allow_dangerous_deserialization=True)
            cls.index = cls.vector_store.index
            logging.info("✅ 벡터 스토어 로드 완료")
        elif os.path.exists(cls.pdf_path):
            logging.error("⚠ 사전 구축된 벡터 스토어를 찾을 수 없습니다. 신규 벡터 스토어를 구축합니다.")
            docs = cls._build_docs()

            texts = [doc.page_content for doc in docs]
            embeddings = cls.embedding_model.embed_documents(texts)

            text_embedding_pairs = list(zip(texts, embeddings))

            cls.vector_store = FAISS.from_embeddings(
                text_embeddings=text_embedding_pairs,
                embedding=cls.embedding_model.embed_query
            )

            cls.vector_store.save_local(cls.vector_store_path)
        else:
            logging.error("❌ PDF를 찾을 수 없습니다. 개발 용어 PDF가 필요합니다.")
            raise FileNotFoundError("Vector store not found. Ensure it is pre-built.")


    @classmethod
    def _build_docs(cls):
        loader = PyPDFLoader(cls.pdf_path)
        pages = loader.load_and_split()

        full_text = "".join([page.page_content for page in pages])

        # 줄바꿈 기준으로 split하는 Text Splitter 생성
        text_splitter = CharacterTextSplitter(
            separator="\n \n",  # 줄 단위로 분할
            chunk_size=500,    # 한 줄씩 개념을 나누도록 설정
            chunk_overlap=0
        )

        # 문서 데이터 분할
        docs = text_splitter.create_documents([full_text])

        for doc in docs:
            doc.page_content = doc.page_content.replace("\n", "")

        return docs


# # 예제 실행
if __name__ == "__main__":
    print("✅ 스크립트 실행 시작")

    logging.basicConfig(level=logging.INFO)  # 로그 설정 (터미널 출력)
    
    retriever = RAGRetriever()  # RAG 검색기 인스턴스 생성
    example_query = "게시판 CRUD 프로그램 만들어줘"  # 검색할 문장
    similar_terms = retriever.search_similar_terms(example_query)  # 검색 실행
    
    print(f"\n🔍 검색 결과:\n{similar_terms}\n")  # 검색 결과 출력
