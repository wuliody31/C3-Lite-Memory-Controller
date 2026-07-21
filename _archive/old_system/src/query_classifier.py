from __future__ import annotations


def classify_query(question: str) -> str:
    q = question.lower()
    if any(x in q for x in ['not enough','explicitly','definitely','exact current','exact price','approved','required','do you know','without evidence','有没有明确','是否明确','不知道','证据不足']):
        return 'abstention'
    if any(x in q for x in ['explain','why','evidence','support','which memory','cite','依据','为什么','解释','证据']):
        return 'explainability'
    if any(x in q for x in [' or ',' rather than ','instead of','which is more current','conflict','supersede','到底','还是','冲突']):
        return 'conflict_resolution'
    if any(x in q for x in ['change','changed','over time','shift','became','originally','previously','later','how did','后来','变化','以前','之前']):
        return 'temporal_update'
    if any(x in q for x in ['how should','style','format','should you','should the system','what should','rewrite','answer structure','应该怎么','格式','风格']):
        return 'procedural_following'
    if any(x in q for x in ['when','before','originally','previously','past','history','什么时候','之前','过去']):
        return 'episodic_recall'
    if any(x in q for x in ['current','now','currently','main','focus','现在','当前','目前']):
        return 'current_fact'
    return 'general'
