from promptflow.connections import AzureOpenAIConnection
from openai import AzureOpenAI
import requests
from lxml import etree
import pandas as pd
import os
import tiktoken
from promptflow.core import tool

class ContentProcessor:
    def __init__(self, connection: AzureOpenAIConnection):
        self.client = AzureOpenAI(
            api_key=connection.api_key,
            api_version=connection.api_version,
            azure_endpoint=connection.api_base
        )
        # 직접 임베딩 모델 deployment 이름 지정
        self.deployment_name = "text-embedding-3-large"  # 또는 실제 사용하는 임베딩 모델의 deployment 이름

    def generate_embeddings(self, text):
        return self.client.embeddings.create(
            input=[text], 
            model=self.deployment_name
        ).data[0].embedding


@tool
def process_csv_content(connection: AzureOpenAIConnection, csv_path: str = "/workspaces/promptflow/klik/data/klik_jd.csv"):
    # ContentProcessor 초기화
    processor = ContentProcessor(connection)
    
    # CSV 파일 읽기
    df = pd.read_csv(csv_path)
    
    # 결과를 저장할 새로운 데이터프레임 생성
    result_df = pd.DataFrame()
    
    # seq 컬럼 저장
    result_df['seq'] = df['seq']
    
    # seq를 제외한 모든 컬럼의 텍스트 결합
    combined_texts = []
    for _, row in df.iterrows():
        # seq 컬럼을 제외한 모든 컬럼 가져오기
        other_columns = [col for col in df.columns if col != 'seq']
        # 해당 컬럼들의 값만 결합
        row_text = ' '.join(str(row[col]) for col in other_columns if pd.notna(row[col]))
        combined_texts.append(row_text)
    
    # 결합된 텍스트를 새로운 컬럼으로 추가
    result_df['combined_text'] = combined_texts
    
    # tiktoken으로 토큰 수 계산
    tokenizer = tiktoken.get_encoding("cl100k_base")
    result_df['n_tokens'] = result_df['combined_text'].apply(
        lambda x: len(tokenizer.encode(x))
    )
    
    # 텍스트 벡터화
    result_df['content_vector'] = result_df['combined_text'].apply(
        lambda x: processor.generate_embeddings(x)
    )
    
    # 결과를 CSV 파일로 저장
    output_file = '/workspaces/promptflow/klik/data/processed_test.csv'
    result_df.to_csv(output_file, index=False)
    
    return {
        "message": f"처리가 완료되었습니다. 결과가 {output_file}에 저장되었습니다.",
        "row_count": len(result_df),
        "sample_row": result_df.iloc[0].to_dict(),
        "output_file": output_file
    }