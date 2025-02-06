from promptflow.core import tool

@tool
def merge_job_text(
    title: str,
    position: str,
    industry: str,
    koreanLevel: str,
    languageLevel: str,
    jobDuties: str,
    preferences: str,
    workCondition: str,
    workLanguage: str
) -> str:
    """
    채용 관련 필드들을 하나의 문장으로 병합하는 함수

    Args:
        title (str): 제목
        position (str): 직무
        industry (str): 산업
        koreanLevel (str): 한국어능력
        languageLevel (str): 언어능력
        jobDuties (str): 담당업무
        preferences (str): 우대사항
        workCondition (str): 근무조건
        workLanguage (str): 노출언어

    Returns:
        str: 병합된 하나의 문장
    """
    
    # 모든 입력값을 리스트로 결합
    values = [
        str(title),
        str(position),
        str(industry),
        str(koreanLevel),
        str(languageLevel),
        str(jobDuties),
        str(preferences),
        str(workCondition),
        str(workLanguage)
    ]
    
    # 빈 문자열이나 None 값 제거
    filtered_values = [v for v in values if v and v.strip()]
    
    # 공백으로 구분하여 하나의 문장으로 결합
    merged_text = " ".join(filtered_values)
    
    return merged_text
