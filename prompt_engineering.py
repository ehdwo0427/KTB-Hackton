import os
import re
import logging
import asyncio

from enum import Enum
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Optional, List, Dict
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# from rag import RAGRetriever

# 환경 변수 로드 (예: API 키)
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# 로그 설정 (오류만 출력)
logging.basicConfig(level=logging.ERROR)

# LangChain LLM (Gemini 모델) 초기화
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GEMINI_API_KEY, temperature=0.3)

# 요청 데이터 모델 정의
class Room(BaseModel):
    id: int
    name: Optional[str] = None  # None 허용하도록 수정
    created_at: Optional[str] = None

class MessageHistory(BaseModel):  # 메시지 히스토리용 새로운 모델
    content: str
    sender: str
    created_at: str

class Message(BaseModel):
    content: str
    role: str

class RequestData(BaseModel):
    room: Room
    message_history: List[MessageHistory]  # 새로운 모델로 변경
    new_message: Message
    type: Optional[str] = "makecode"

# 코드 생성기를 담당하는 클래스
class CodeGenerator:
    """Python 코드 생성기 (RAG 미적용)"""

    @classmethod
    # 비동기 함수: 코드 생성 실행 및 결과 파일 저장
    async def run_code_generation(cls, request: RequestData):
        """
        코드 생성 요청을 실행하고,
        생성된 Python 코드를 파일과 MD 형식의 설명 파일로 저장하며 출력하는 함수.
        """
        # 비동기적으로 코드 생성 수행
        result = await CodeGenerator.generate_code(request)
        print(result)

        code_text, readme_text = result.split("---")
        readme_text = readme_text.strip()

        if '불가능' in code_text:
            return "Sub_question", readme_text
        
        # 원하는 폴더 경로 설정 (서버의 특정 폴더)
        base_folder_path = "/root/docker/generate_projects"

        folder_list = []
        
        # 폴더가 존재하지 않으면 생성
        if not os.path.exists(base_folder_path):
            os.makedirs(base_folder_path)

        # 프로젝트 폴더 개수 확인 후 새로운 프로젝트 폴더 번호 할당
        existing_projects = [name for name in os.listdir(base_folder_path) if os.path.isdir(os.path.join(base_folder_path, name))]
        project_counter = len(existing_projects) + 1
        project_folder_path = os.path.join(base_folder_path, f"project{project_counter}")

        # 새 프로젝트 폴더 생성
        os.makedirs(project_folder_path)

        
        # 마크다운 블록을 파싱하여 파일별로 저장
        md_pattern = re.compile(r"```(?:python)?\s*(.*?)\s*```", re.DOTALL)
        files_text = md_pattern.findall(code_text)

        # print(files_text)
        # 파일 저장
        for file_text in files_text:
            name_pattern = re.compile(r"# ([^\n]+)\n([\s\S]*)", re.DOTALL)
            code_match = re.search(name_pattern, file_text)

            content, filename = code_match.group(0), code_match.group(1)
           
            # Python 코드 파일 경로
            code_save_path = os.path.join(project_folder_path, filename)
            directory = os.path.dirname(code_save_path)  # 폴더 경로만 추출

            # 🔹 폴더가 없으면 생성 (이미 존재하면 무시)
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
                
            folder_list.append(code_save_path)
            # 생성된 Python 코드를 지정된 폴더에 "generated_code.py" 파일로 저장
            with open(code_save_path, "w", encoding="utf-8") as py_file:
                py_file.write(content)
        
        
        # Description 파일 경로
        readme_save_path = os.path.join(project_folder_path, 'README.md')

        # 생성된 설명 메시지를 지정된 폴더에 md 형식으로 저장
        with open(readme_save_path, "w", encoding="utf-8") as md_file:
            md_file.write(readme_text)
        folder_list.append(readme_save_path)

        return "project_folder_list", folder_list

    @classmethod
    async def generate_code(cls, request: RequestData, model: str = "gemini-1.5-flash") -> dict:
        """
        비동기 방식으로 Gemini API를 호출하여 코드를 생성하는 함수.
        1. 요청 정보를 바탕으로 프롬프트 생성
        2. LLM 호출하여 응답 받기
        3. 응답을 코드 부분과 설명 부분으로 분리
        4. 코드 부분의 오류를 검증 및 수정하여 최종 코드를 반환
        """
        # 요청 정보를 바탕으로 프롬프트 생성
        prompt = cls._generate_prompt(request)
        # LLM을 비동기적으로 호출 (동기 함수를 executor로 실행)
        response = await asyncio.get_event_loop().run_in_executor(
            None, lambda: llm.invoke(prompt)
        )
        # LLM의 응답에서 content를 추출 (없으면 "코드 생성 실패" 메시지)
        full_response = response.content if hasattr(response, 'content') else "코드 생성 실패"
        return full_response

    @classmethod
    def _generate_prompt(cls, request: RequestData) -> str:
        """
        LangChain의 PromptTemplate을 사용하여 최적화된 프롬프트를 생성하는 함수.
        요청에 포함된 설명, 코드 스타일, 주석 포함 여부, 코드 구조 정보를 템플릿에 채워서 반환.
        """
        # rag_prompt = RAGRetriever.search_similar_terms(request.description)

        message_history = ""

        for message_hist in request.message_history:
            content, sender, created_at = message_hist
            message_history += sender + " : " + content + "\n"
        
        message_history += request.new_message.role + " : " + request.new_message.content + "\n"



        template = PromptTemplate(
            input_variables=["message_history"],
            template="""
            너는 Python 코드 생성을 전문으로 하는 AI야.
            사용자 입력에 해당하는 기능을 구현해야 해.

            사용자 입력:
            "{message_history}"

            작업 가능 여부 판단
            사용자 입력이 개발 기능을 구체적으로 설명하는 지 판단해서 답변해야 해. Case에 맞게 답변 양식을 정확히 따라야 해.
            Case 1️⃣ **기능을 구체적으로 설명하는 경우**
                - 사용자 입력이 개발하고자 하는 명확한 기능을 포함하고 있어야 해.
                - 직관적인 기능의 MVP를 개발 가능한 입력이여야 해.
                - **답변 양식**
                ```
                [하단의 기능 구현 내용 출력]
                ```
            Case 2️⃣ **기능울 구체적으로 설명하지 않는 경우**
                - 사용자 입력만으로 개발하고자 하는 명확한 기능을 판단할 수 없어야 해.
                - 기능을 개발하는 동안 사용자에게 추가 질문이 필요한 경우여야 해.
                - 추가 질문은 3개 이내의 선택지 제공을 통해 진행해야해.
                - **답변 양식**
                ```
                # 개발 불가능
                ---
                [사용자에게 제공할 추가 질문 출력]
                ```

            
            **기능 구현**
            기능을 구현하기 위해 아래 작업 순서를 반드시 따라야 해.

            작업 순서
            1️⃣ **프로젝트 폴더 구조 설계**
                - root 디렉토리를 기반으로 해당 기능을 배포할 수 있는 전체 코드 구조를 우선적으로 설계해야 해.
                - 설계된 프로젝트 폴더 구조는 가장 먼저 출력해야 해.
            2️⃣ **각 파일 별 코드 구현**
                - 각 파일에 해당하는 기능의 코드를 구현해야 해.
                - 파일 별로 markdown 코드 블록(```python ... ```) 안에 파일 경로, 코드 구조를 출력해야 해.
                - Python 코드 뿐 만 아니라 배포에 필요한 환경 설정 파일(requirements.txt 등)도 명확하게 출력해야 해.
            3️⃣ **실행 순서에 따른 코드 설명**
                - 전체 코드 구조 출력이 끝난 후에 **코드 설명**을 출력해야 해.
                - 코드 설명은 **파일 별 설명**과 **배포 작업 순서 설명**으로 구성해야 해.
                - **배포 작업 순서 설명**을 순서대로 진행하면 정확한 기능이 배포되어야 해.
            

            이 부분은 매우 중요해. Python 코드를 줄 때 반드시 이 형식을 지켜야 해!!!
            사용자가 요청한 대로 코드가 올바르게 실행될 수 있도록 코드를 작성해야 해.
            
            🛠️ 필수 요구 사항(일관성)
            동일한 사용자 입력에 대해 항상 동일한 코드와 설명을 출력해야 해.  
            무작위성이 개입되지 않도록 결정론적으로 작성해야 해.

            🛠️ 필수 요구 사항(정확성)
            모든 Python 코드는 문법 오류, 런타임 오류 없이 실행 가능해야 해.
            각 파일 별 코드가 독립적이며, 참조 및 import 오류가 발생하지 않도록 정확한 라이브러리 명칭과 코드 구성이 필요해.
            작성된 코드가 실제 기능을 완벽히 수행하도록 검증된 코드를 생성할 것.

            🛠️ 필수 요구 사항(설계)
            사용자 입력에 배포 프레임워크가 특정되어 있지 않으면 **FastAPI**를 활용한 폴더 구조를 설계해야 해.
            사용자 입력에 DB 프레임워크가 특정되어 있지 않으면 **SQLite**를 활용한 DB를 설계해야 해. SQLAlchemy는 사용하면 안돼.
            requirements.txt 작성 시 버전 충돌로 인한 오류(AssertionError)가 발생하지 않도록 정확하게 작성해야 해.
            requirements.txt 미완성으로 인한 참조 오류(ImportError)가 발생하지 않도록 정확하게 작성해야 해.
            프로젝트 폴더 구조가 배포 과정을 전부 소화할 수 있도록 프로젝트 폴더 구조를 꼼꼼하고 명확하게 설계해야 해.
            전체 배포 과정이 터미널만 사용해서 이루어질 수 있도록 설계해야 해.
            
            🛠️ 필수 요구 사항(코드)
            PEP8 스타일로 작성해야 해.
            Python 문법 오류(SyntaxError)가 없어야 해.
            실행 시 런타임 오류(RuntimeError)가 발생하지 않아야 해.
            각 파일 별 기능을 참조할 시 오류(ImportError, ModuleNotFoundError)가 발생하지 않아야 해.
            정확한 라이브러리 명을 참조해서 오류(ImportError, ModuleNotFoundError)가 발생하지 않아야 해.
            클래스 매서드와 인스턴스 매서드의 차이를 명확히 인지하고 오류가 발생하지 않게 사용해야 해.
            코드의 논리는 정확해야 하며, 기능은 완벽히 작동해야 해.
            End 사용자가 바로 사용할 수 있도록 배포 해줘야 해.
            백점 만점의 답변을 제공해줘야 해.

            🛠️ 필수 요구 사항(배포)
            비동기 함수 사용 시 통신 과정에서 오류가 발생하지 않아야 해.
            통신 과정에서의 입력, 출력 형식이 정확하게 정의되어서 오류가 발생하지 않아야 해. 예를 들어 CRUD 기능을 구현할 때 각각의 입력과 출력이 오류가 발생하지 않게 정의되어야 해.
            Pydantic Basemodel을 활용하여 배포 과정에서의 입력을 명확하게 표기해주어야 해.

            📌 📢 중요한 출력 형식 요구 사항
            아래 출력 형식을 반드시 따라야 해.
            파일 단위의 markdown 코드 블록 출력 형식을 반드시 따라야 해!
            대괄호 안에 있는 변수를 각 출력으로 채워야 해.
            배포 작업 순서에 폴더를 구축하는 내용은 들어가면 안돼. 그 이후부터 작성해야 해.
            파일 이름에는 설명이 붙지 않아야 해.

            **출력 형식**            
            ```python
            # [파일 이름]
            [코드]
            ```

            ```python
            # [파일 이름]
            [코드]
            ```

            ```python
            # [파일 이름]
            [코드]
            ```

            ---

            # [프로젝트 이름]
            [간략한 프로젝트 설명]

            ## 폴더 구조
            [폴더 구조]

            ## 파일 별 설명
            - [파일 이름] : [파일 설명]
            - [파일 이름] : [파일 설명]
            

            ## 배포 작업 순서 설명
            1. [작업 설명]
            2. [작업 설명]

            🎯 코드 생성 요청: 이제 Python 코드와 설명을 생성해. 설명은 한국어로 작성해야 해.
            """
        )
        # 템플릿에 요청 정보를 채워 최종 프롬프트 생성
        return template.format(
            message_history=message_history
            # rag_prompt=rag_prompt
        )