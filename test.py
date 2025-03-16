from langchain import PromptTemplate, LLMChain
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import os

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# Gemini 모델과의 통합을 위한 가상의 LLM 클래스 (실제 Gemini 연동 모듈로 대체)
class GeminiLLM:
    def __init__(self, api_key: str):
        self.api_key = api_key
        # 실제 초기화 코드 추가

    def __call__(self, prompt: str) -> str:
        # 실제 Gemini API 호출 코드 구현
        # 예시로 고정 응답 반환 (실제 환경에서는 API 결과 반환)
        return "단계 1: 문제 분석 결과 ...\n단계 2: 해결 전략 ...\n단계 3: 결론 도출 ..."

# 프롬프트 템플릿 정의
template = """
문제: {problem}

아래 단계에 따라 문제를 해결하라:
1. 문제를 분석하라.
2. 가능한 해결 전략을 제시하라.
3. 결론을 도출하라.

응답은 각 단계를 순서대로 서술할 것.
"""

prompt = PromptTemplate(input_variables=["problem"], template=template)

# Gemini 모델 초기화 (API 키는 실제 값으로 대체)
gemini = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=GEMINI_API_KEY)

# LLM 체인 구성
chain = LLMChain(llm=gemini, prompt=prompt)

# 체인 실행 예시
problem_statement = "XOR 게이트를 코드로 전달해줘"
result = chain.run(problem=problem_statement)

print("모델 응답:")
print(result)
