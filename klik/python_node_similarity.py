from promptflow import tool
from typing import Dict, Any
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity



@tool
def calculate_similarity(input1: Dict[str, Any], input2: Dict[str, Any]) -> Dict[str, Any]:
    # input1에서 비교할 벡터 가져오기
    content_vector1 = np.array(eval(input1.get("content_vector"))) if isinstance(input1.get("content_vector"), str) else np.array(input1.get("content_vector"))
    
    # input2에서 파일 경로 가져오기
    output_file = input2.get("output_file")
    
    # CSV 파일 읽기
    df = pd.read_csv(output_file)
    
    # 벡터 문자열을 numpy 배열로 변환
    df['content_vector'] = df['content_vector'].apply(eval)
    vectors = np.array(df['content_vector'].tolist())
    
    # 한 번에 모든 벡터와의 유사도 계산
    similarities = cosine_similarity([content_vector1], vectors)[0]
    
    # 가장 유사도가 높은 행 찾기
    most_similar_idx = np.argmax(similarities)
    most_similar_row = df.iloc[most_similar_idx]
    max_similarity = similarities[most_similar_idx]
    
    # 결과 반환
    #return {
    #    "most_similar_content": most_similar_row[['combined_text', 'n_tokens']].to_dict(),
    #    "similarity_score": float(max_similarity)
    #    #"original_vector": content_vector1.tolist()
    #}


    # 입력값과 가장 유사한 공고 정보를 결합
    required_fields = {
        "title": input1.get("title"),
        "position": input1.get("position"),
        "industry": input1.get("industry"),
        "koreanLevel": input1.get("koreanLevel"),
        "languageLevel": input1.get("languageLevel"),
        "jobDuties": input1.get("jobDuties"),
        "preferences": input1.get("preferences"),
        "workCondition": input1.get("workCondition"),
        "workLanguage": input1.get("workLanguage")
    }

    return {
        "most_similar_content": most_similar_row[['combined_text', 'n_tokens']].to_dict(),
        "similarity_score": float(max_similarity),
        "original_vector": content_vector1.tolist(),
        "required_fields": required_fields,
        "prompt_instruction": "다음의 필수 포함 사항들을 반드시 활용하여 유사 공고를 참고해 새로운 공고를 작성해주세요:"
    }