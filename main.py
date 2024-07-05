from fastapi import FastAPI, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from module_friendship import (
    analyze_sentiments, organize_dialogues, parse_dialogues, 
    calculate_percentage_scores, calculate_friendship,
    rule1, rule2, rule3, filter_emotions, compare_scores,
    narration_emotion_changes
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/friendship")
async def upload_file(file: UploadFile):
    try:
        contents = await file.read()
        lines = contents.decode('utf-8').splitlines()

        # 로그 추가
        #print("Uploaded file contents:", lines)

        result = parse_dialogues(lines)
        dialogues, combined_dialogues = organize_dialogues(result)
        resultOk = 'y'
        if len(dialogues) != 2:
            resultOk = 'n'
            return resultOk
        else:
            names, score, scoreList, mixed_results, sentiment_avg_scores, check_score, scoreList2 = analyze_sentiments(dialogues, combined_dialogues)

            # 백분율 계산
            sentiment_avg_scores_percentage = calculate_percentage_scores(sentiment_avg_scores)

            # 우정도 계산
            friendship_scores = {name: calculate_friendship(scores) for name, scores in sentiment_avg_scores.items()}

            # 우선순위 예시 사용법
            filtered_results = filter_emotions(sentiment_avg_scores)

            #누가 누구를 더 좋아한다
            compare = compare_scores(check_score)

            #나레이션
            narration = narration_emotion_changes(scoreList2)

            #기준1 - 누가 총을 대신 맞아줄 것인가?
            emotions_of_interest = ["고마운", "설레는(기대하는)", "기쁨(행복한)"]
            rules1 = rule1(sentiment_avg_scores_percentage, emotions_of_interest)

            #기준2 - 흔들리지 않는 편안한 침대 같은 사람
            emotions_of_interest1 = ["일상적인", "즐거운(신나는)"]
            rules2 = rule2(sentiment_avg_scores_percentage, emotions_of_interest1)

            #기준3 - 뒷 통수 칠 사람
            emotions_of_interest2 = ["짜증남", "걱정스러운(불안한)", "힘듦(지침)", "생각이 많은", "걱정스러운(불안한)"]
            rules3 = rule3(sentiment_avg_scores_percentage, emotions_of_interest2)
            
            result = {
                #개별 감정 결과, 이름, 시간, 채팅, 감정, 스코어, 누적 변화량
                "individual_results": names,
                
                #그래프로 표현할 누적 변화량
                "individual_score_lists_for_graph": scoreList2,
                
                #사용자별 감정 평균 점수
                "sentiment_avg_scores": sentiment_avg_scores,
                
                #사용자별 감정 평균 점수 백분율로 표시
                "sentiment_avg_scores_percentage": sentiment_avg_scores_percentage,
                
                #각 사용자의 감정 평균 점수의 총합 - 최대 100
                "individual_scores": check_score,
                
                #최종 우정점수, 결과페이지 ℃표현할 때 사용할 점수
                "friendship_scores": friendship_scores,

                "resultOk": resultOk,
                
                #우선순위
                "Rankings" : filtered_results,
                
                #누가 누구를 더 좋아한다
                "more_friendly" : compare,
                
                #나레이션
                "narration" : narration,
                
                #기준1 - 누가 총을 대신 맞아줄 것인가?
                "gun" : rules1,
                
                #기준2 - 흔들리지 않는 편안한 침대 같은 사람
                "bed" : rules2,
                
                #기준3 - 뒷 통수 칠 사람
                "betrayer" : rules3
            }
        #print("Final Result in main.py:", result)  # 최종 결과 로그 추가

        return result

    except Exception as e:
        print("Error in main.py:", str(e))  # 오류 로그 추가
        return {"error": str(e)}

#uvicorn main:app --reload --port 8000
