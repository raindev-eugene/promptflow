from promptflow.core import tool
from promptflow.connections import AzureOpenAIConnection
from openai import AzureOpenAI
import tiktoken

class ContentProcessor:
    def __init__(self, connection: AzureOpenAIConnection):
        self.client = AzureOpenAI(
            api_key=connection.api_key,
            api_version=connection.api_version,
            azure_endpoint=connection.api_base
        )
        self.deployment_name = "text-embedding-3-large"

    def generate_embeddings(self, text):
        return self.client.embeddings.create(
            input=[text], 
            model=self.deployment_name
        ).data[0].embedding

@tool
def process_merged_text(merged_text: str, connection: AzureOpenAIConnection) -> dict:
    """
    병합된 텍스트에 대해 토큰 수를 계산하고 임베딩을 생성하는 함수

    Args:
        merged_text (str): text_merger.py에서 생성된 병합된 텍스트
        connection (AzureOpenAIConnection): Azure OpenAI 연결 객체

    Returns:
        dict: 원본 텍스트, 토큰 수, 임베딩 벡터를 포함하는 딕셔너리
    """
    
    # 토큰 수 계산
    tokenizer = tiktoken.get_encoding("cl100k_base")
    n_tokens = len(tokenizer.encode(merged_text))
    
    # 임베딩 생성
    processor = ContentProcessor(connection)
    content_vector = processor.generate_embeddings(merged_text)
    
    return {
        "content": merged_text,
        "n_tokens": n_tokens,
        "content_vector": content_vector
    }