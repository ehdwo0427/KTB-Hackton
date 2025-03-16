import os
import re
import io
import sys
import ast
import logging
import asyncio
import warnings
import traceback

from enum import Enum
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain.prompts import PromptTemplate
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# from vector_store_db import check_vector_store, save_to_vector_store
# from vector_search import search_similar_code

# 환경 변수 로드
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# 로그 설정
logging.basicConfig(level=logging.DEBUG)

# LangChain LLM 초기화
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=GEMINI_API_KEY)

class CodeStyle(str, Enum):
    PEP8 = "PEP8"
    Google = "Google"
    NoneStyle = "None"

class CodeStructure(str, Enum):
    Functional = "functional"
    ClassBased = "class-based"

class CodeRequest(BaseModel):
    description: str
    style: CodeStyle = CodeStyle.PEP8
    include_comments: bool = False
    structure: CodeStructure = CodeStructure.Functional

class CodeGenerator:
    """Python 코드 생성기 (RAG 미적용)"""

    @classmethod
    async def generate_code(cls, request: CodeRequest, model: str = "gemini-pro") -> str:
        """비동기 Gemini API 호출 (RAG 미적용)"""
        
        # # 기존 코드 확인 (Vector Store)
        # cached_code = check_vector_store(request.description)
        # if cached_code:
        #     return cached_code

        # # 유사한 코드 검색 (RAG)
        # similar_codes = search_similar_code(request.description, top_k=1)
        # similar_code_text = similar_codes[0][1] if similar_codes else "참고 코드 없음"

        prompt = cls._generate_prompt(request)

        logging.warning(f"프롬프트 : {prompt}")

        # LangChain LLM 호출
        response = await asyncio.get_event_loop().run_in_executor(
            None, lambda: llm.invoke(prompt)
        )

        generated_code = response.content if isinstance(response, AIMessage) else "코드 생성 실패"

        # 🔹 마크다운 코드 블록 제거
        cleaned_code = cls._remove_markdown_code_blocks(generated_code)

        # 🔹 코드 후처리 실행 (문법 및 실행 오류 검사)
        validated_code = cls._validate_and_fix_code_until_no_error(cleaned_code)
        
        ### 코드 저장
        

        # # Vector Store에 저장 (DB 활용)
        # save_to_vector_store(request.description, generated_code)

        return validated_code

    @classmethod
    def _generate_prompt(cls, request: CodeRequest) -> str:
        """LangChain PromptTemplate을 사용한 최적화된 프롬프트 생성"""

        # 변환된 문자열 변수 정의 (일관된 방식 적용)
        include_comments_text = "포함" if request.include_comments else "제외"
        structure_text = "함수형" if request.structure == "functional" else "클래스형"
        add_vector_store_text = ""

        template = PromptTemplate(
            input_variables=["description", "style", "include_comments", "structure", "similar_code", "add_vector_store"],
            template="""
            너는 Python 코드 생성을 전문으로 하는 AI야.
            사용자가 요청한 코드가 **올바르게 실행될 수 있도록** 작성해야 해.
            

            ### 🛠️ 필수 조건
            - Python 문법 오류(SyntaxError)가 없어야 함.
            - 실행 시 오류(RuntimeError)가 발생하지 않아야 함.
            - 코드의 논리가 정확해야 하며, 예상된 출력이 나와야 함.
            


            ### 🎨 코드 스타일 & 구조
            - 코드 스타일: {style}
            - 주석 포함 여부: {include_comments}
            - 코드 구조: {structure}

            ### 📌 📢 중요한 출력 형식 요구사항
            - **출력된 코드는 그대로 실행 가능한 상태여야 하며, 코드 시작과 끝에 불필요한 텍스트가 포함되지 않도록 할 것.**
            - **예제 코드가 필요한 경우, `#`을 사용한 Python 주석으로 추가할 것.**
            - **100점 만점의 답변으로 평가됩니다.** 
            - 기존 템플릿에 출력 형식 관련 조건을 추가하여 두 섹션(코드와설명)으로 명시합니다.

            ### 🎯 코드 생성 요청
            "{description}"
            
            ### 추가 고려사항
            "{add_vector_store}"
            """
        )

        return template.format(
            description=request.description,
            style=request.style.value,
            include_comments=include_comments_text,  
            structure=structure_text,
            add_vector_store=add_vector_store_text
        )
    
    @staticmethod
    def _remove_markdown_code_blocks(code: str) -> str:
        """마크다운 코드 블록 제거 (```python, ```)"""
        cleaned_code = re.sub(r"```(python)?\n?", "", code)  # 첫 번째 마크다운 제거
        cleaned_code = re.sub(r"```\n?", "\n", cleaned_code)  # 마지막 마크다운 제거, 개행 유지

        return cleaned_code.strip()  # 🔹 앞뒤 공백 제거
    
    @classmethod
    def _validate_and_fix_code_until_no_error(cls, code: str, max_attempts: int = 5) -> str:
        """코드가 오류가 없을 때까지 반복 검사 + 오류 메시지 기반 RAG 적용"""
        error_messages = []  # 🔹 이전 오류 메시지 저장
        for attempt in range(max_attempts):
            syntax_error = cls._check_syntax_error(code)
            runtime_error, execution_output = cls._execute_and_capture_output(code)

            if not syntax_error and not runtime_error:
                return code  # 🔹 오류 없음 → 최종 코드 반환
            
            error_message = f"Attempt {attempt+1} 오류 발생:\n"
            if syntax_error:
                error_message += f"Syntax Error: {syntax_error}\n"
            if runtime_error:
                error_message += f"Runtime Error: {runtime_error}\n"
            if execution_output:
                error_message += f"실행 출력: {execution_output}\n"

            logging.warning(f"⚠️ {error_message.strip()}")
            error_messages.append(error_message)  # 🔹 이전 오류 메시지 저장
            code = cls._fix_code_with_llm(code, error_messages)

        return "코드 수정 실패"

    @staticmethod
    def _check_syntax_error(code: str) -> str:
        """Python 문법 오류 검사"""
        try:
            ast.parse(code)
            return None  # 문법 오류 없음
        except SyntaxError as e:
            return f"{e.msg} (파일: {e.filename}, 라인: {e.lineno}, 컬럼: {e.offset})"

    @staticmethod
    def _execute_and_capture_output(code: str) -> tuple:
        """코드를 실행하고 실행 오류를 감지"""
        captured_output = io.StringIO()
        captured_error = io.StringIO()

        sys.stdout = captured_output  # 표준 출력 리디렉션
        sys.stderr = captured_error  # 표준 에러 리디렉션

        logging.warning(f"코드 :  {code}")
        
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            try:
                exec(code, globals())  # 🔹 실행 환경을 실제 환경과 유사하게 설정
                execution_output = captured_output.getvalue()
                execution_error = captured_error.getvalue()

                logging.warning("✅ 실행 완료, 출력 결과:\n" + execution_output)
                if execution_error:
                    logging.error("⚠️ 실행 중 오류 발생 (stderr):\n" + execution_error)

                return None, captured_output.getvalue()  # 실행 오류 없음
            except ValueError as ve:
                error_traceback = traceback.format_exc()
                logging.error(f"❌ [ValueError] {ve}\n{error_traceback}")
                return f"[ValueError] {ve}\n{error_traceback}", captured_output.getvalue()
            except TypeError as te:
                error_traceback = traceback.format_exc()
                logging.error(f"❌ [TypeError] {te}\n{error_traceback}")
                return f"[TypeError] {te}\n{error_traceback}", captured_output.getvalue()
            except IndexError as ie:
                error_traceback = traceback.format_exc()
                logging.error(f"❌ [IndexError] {ie}\n{error_traceback}")
                return f"[IndexError] {ie}\n{error_traceback}", captured_output.getvalue()
            except KeyError as ke:
                error_traceback = traceback.format_exc()
                logging.error(f"❌ [KeyError] {ke}\n{error_traceback}")
                return f"[KeyError] {ke}\n{error_traceback}", captured_output.getvalue()
            except ZeroDivisionError as zde:
                error_traceback = traceback.format_exc()
                logging.error(f"❌ [ZeroDivisionError] {zde}\n{error_traceback}")
                return f"[ZeroDivisionError] {zde}\n{error_traceback}", captured_output.getvalue()
            except Warning as w:
                error_traceback = traceback.format_exc()
                logging.error(f"⚠️ [Warning] {w}\n{error_traceback}")
                return f"[Warning] {w}\n{error_traceback}", captured_output.getvalue()
            except Exception as e:
                error_traceback = traceback.format_exc()
                logging.error(f"❌ [Unknown Error] {e}\n{error_traceback}")
                return f"[Unknown Error] {e}\n{error_traceback}", captured_output.getvalue()
            finally:
                sys.stdout = sys.__stdout__  # 표준 출력 복원
                sys.stderr = sys.__stderr__  # 표준 에러 복원
    
    @classmethod
    def _fix_code_with_llm(cls, code: str, error_messages: list) -> str:
        """LLM을 사용하여 코드 수정 (오류 메시지 기반 RAG 적용)"""

        # 🔹 누적된 오류 메시지 포함
        error_context = "\n".join(error_messages)

        prompt = f"""
        ### 🔍 Python 코드 오류 수정 요청
        아래 코드에서 문법 및 실행 오류를 수정해줘.

        ### 📌 수정 목표:
        1. 코드가 실행될 때 **문법 오류(SyntaxError)**가 발생하지 않아야 함.
        2. 실행 중 **RuntimeError(ZeroDivisionError, IndexError, TypeError 등)**가 발생하지 않아야 함.
        3. 기존 코드의 논리 구조를 최대한 유지하면서, 오류를 해결할 것.

        ### 📌 📢 중요한 출력 형식 요구사항
        - **출력된 코드는 그대로 실행 가능한 상태여야 하며, 코드 시작과 끝에 불필요한 텍스트가 포함되지 않도록 할 것.**
        - **예제 코드가 필요한 경우, `#`을 사용한 Python 주석으로 추가할 것.**

        ### 📝 이전 오류 메시지
        {error_context}

        ### 🎯 코드 수정 요청
        ```python
        {code}
        ```
        """
        response = llm.invoke(prompt)

        generated_code = response.content if hasattr(response, 'text') else "코드 수정 실패"

        cleaned_code = cls._remove_markdown_code_blocks(generated_code)

        return cleaned_code

#################################
#################################
############ main.py ############
#################################
#################################
import asyncio

# 이미 작성된 코드에 포함된 클래스 및 Enum을 import합니다.
# 만약 별도의 모듈로 분리되어 있다면, 예: from code_generator import CodeGenerator, CodeRequest, CodeStyle, CodeStructure
# 아래 코드는 같은 파일 내에 있다고 가정합니다.

async def main():
    # 예시 코드 생성 요청: 두 숫자를 입력받아 덧셈 결과를 출력하는 함수를 생성하도록 요청
    description = "AND 모델을 퍼셉트론을 써서 설명해줘"
    
    request = CodeRequest(
        description=description,
        style=CodeStyle.PEP8,           # 코드 스타일: PEP8
        include_comments=True,          # 주석 포함 여부
        structure=CodeStructure.Functional  # 함수형 구조
    )
    
    # CodeGenerator를 사용해 비동기로 코드 생성
    generated_code = await CodeGenerator.generate_code(request)
    with open("output.txt", "w", encoding="utf-8") as f:
        f.write(generated_code)
    
    print("=== 생성된 코드 ===")
    print(generated_code)

if __name__ == "__main__":
    # 비동기 메인 함수를 실행
    asyncio.run(main())