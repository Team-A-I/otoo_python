from collections import defaultdict # defaultdict를 사용하여 딕셔너리를 초기화
from datetime import datetime
import re  # 정규 표현식 라이브러리를 가져옴
from transformers import pipeline

model="nlp04/korean_sentiment_analysis_kcelectra"

classifier = pipeline("sentiment-analysis", model="nlp04/korean_sentiment_analysis_kcelectra")

# 데이터 가공 작업 1 - 카카오톡 데이터 파일 내용 분리
def parse_dialogues(lines):
    result = []
    current_date = None
    date_pattern = re.compile(r'--------------- (\d{4}년 \d{1,2}월 \d{1,2}일 [가-힣]+) ---------------')
    message_pattern = re.compile(r'\[(.*?)\] \[(오전|오후) (\d{1,2}:\d{2})\] (.*)')

    for line in lines:
        date_match = date_pattern.match(line)
        if date_match:
            current_date = date_match.group(1)
            continue
        
        message_match = message_pattern.match(line)
        if message_match:
            user, am_pm, time, message = message_match.groups()
            if current_date:
                result.append(f'[{user}] [{current_date} {am_pm} {time}] {message}')
            else:
                result.append(f'[{user}] [{am_pm} {time}] {message}')
        else:
            result.append(line.strip())
    return result

# 데이터 가공 작업 2 - 분리된 내용 새로운 형태로 정의
def organize_dialogues(parsed_lines):
    pattern = re.compile(r'\[(.*?)\] \[(.*?)\] (.*)')
    dialogues = defaultdict(list)
    combined_dialogues = []
    
    current_name = None
    current_time = None
    current_message = None

    for line in parsed_lines:
        if not line.strip():
            continue
        match = pattern.match(line)
        if match:
            current_name, current_time, current_message = match.groups()
            dialogues[current_name].append((current_name, current_time, current_message))
            combined_dialogues.append((current_name, current_time, current_message))
        else:
            if current_name and current_time and current_message is not None:
                updated_message = dialogues[current_name][-1][2] + " " + line.strip()
                dialogues[current_name][-1] = (current_name, current_time, updated_message)
                combined_dialogues[-1] = (current_name, current_time, updated_message)
    
    return dialogues, combined_dialogues


# 감정 분석
def analyze_sentiments(dialogues, combined_dialogues):
    sentiment_scores = defaultdict(lambda: defaultdict(list))
    sentiment_message_count = {}
    names = {}
    score = {}
    sumscore = {}
    scoreList = {}
    sumscore2 = {}
    scoreList2 = {}

    for name, dialogues_list in dialogues.items():
        count = 0
        names[name] = []
        score[name] = 0.0
        sumscore[name] = 50
        scoreList[name] = []
        sumscore2[name] = 50
        scoreList2[name] = []

        for dialogue in dialogues_list:
            result = classifier(dialogue[2])[0]
            count += 1
            sentiment_scores[name][result['label']].append(result['score'])

            if result['label'] in ['고마운', '기쁨(행복한)', '즐거운(신나는)', '일상적인', '설레는(기대하는)']:
                score[name] += 1
                sumscore[name] += 1
            scoreList[name].append(sumscore[name])
            names[name].append((dialogue[0], dialogue[1], dialogue[2], result['label'], result['score'], sumscore[name]))

        score[name] = score[name] / count * 100
        sentiment_message_count[name] = count

    sentiment_avg_scores = {name: {sentiment: round(sum(scores) / sentiment_message_count[name] * 100) 
                                   for sentiment, scores in sentiments.items()} 
                            for name, sentiments in sentiment_scores.items()}
    
    check_score = {s: sum(sentiment_avg_scores[s].values()) for s in sentiment_avg_scores}

    all_names = list(dialogues.keys())
    mixed_results = []
    for dialogue in combined_dialogues:
        result = classifier(dialogue[2])[0]
        # print(f"dialogue[0]:{dialogue[0]}, dialogue[1]:{dialogue[1]}, dialogue[2]:{dialogue[2]}")
        mixed_results.append(dialogue[2])

        if dialogue[0] == all_names[0]:
            if result['label'] in ['고마운', '기쁨(행복한)', '즐거운(신나는)', '일상적인', '설레는(기대하는)']:
                sumscore2[all_names[0]] += 1
            scoreList2[all_names[0]].append(sumscore2[all_names[0]])
            scoreList2[all_names[1]].append(sumscore2[all_names[1]])
        elif dialogue[0] == all_names[1]:
            if result['label'] in ['고마운', '기쁨(행복한)', '즐거운(신나는)', '일상적인', '설레는(기대하는)']:
                sumscore2[all_names[1]] += 1
            scoreList2[all_names[0]].append(sumscore2[all_names[0]])
            scoreList2[all_names[1]].append(sumscore2[all_names[1]])

    # print(f"len(mixed_results){len(mixed_results)}")
    return names, score, scoreList, mixed_results, sentiment_avg_scores, check_score, scoreList2


# 감정스코어 평균합산에 대한 백분율을 만드는 코드
def calculate_percentage_scores(sentiment_avg_scores):
    percentage_scores = {}  # 백분율 스코어를 저장할 딕셔너리 초기화
    
    # 각 사용자의 감정 점수를 순회
    for name, sentiments in sentiment_avg_scores.items():
        total_score = sum(sentiments.values()) # 모든 감정 점수의 합을 계산
        percentage_scores[name] = {sentiment: round((score / total_score) * 100, 2) for sentiment, score in sentiments.items()}  # 각 감정에 대해 백분율을 계산하여 딕셔너리에 저장
    return percentage_scores # 백분율 스코어를 반환


# 우정도 수치 계산
# 각 감정의 점수에 가중치를 부여 후에 그 가중치에 곱한 후 합산
# 총 대화 수로 나누어 평균 우정도를 구한다.
def calculate_friendship(sentiment_scores):
    # 감정 범주에 따라 가중치를 설정합니다.
    weights = {
        '슬픔(우울한)': -0.5,
        '짜증남': -0.5,
        '생각이 많은': -0.5,
        '걱정스러운(불안한)': -0.5,
        '힘듦(지침)': -0.5,
        '일상적인': 1,
        '즐거운(신나는)': 1,
        '기쁨(행복한)': 1,
        '설레는(기대하는)': 1,
        '고마운': 1,
        '사랑하는': 0
    }
    
    total_weighted_score = 0  # 총 가중치 점수를 저장할 변수 초기화
    total_count = 0  # 총 점수를 저장할 변수 초기화
    
    # 각 감정과 그 점수를 순회
    for emotion, score in sentiment_scores.items():
        weight = weights.get(emotion, 0) # 해당 감정의 가중치를 가져옴 (기본값은 0)
        total_weighted_score += score * weight # 감정 점수에 가중치를 곱해 총 가중치 점수에 더함
        total_count += score # 총 점수에 현재 점수를 더함
    
    if total_count == 0:
        return 0 # 총 점수가 0이면 0을 반환
    
    # 호감도 점수를 0에서 1 사이의 값으로 정규화
    friendship_score = (total_weighted_score / total_count + 1) / 2

    return friendship_score  # 계산된 호감도 점수를 반환


def convert_to_24h_time(am_pm, time):
    try:
        # 시간과 분을 ':' 기준으로 분리하고 정수형으로 변환
        hour, minute = map(int, time.split(':'))
        
        # '오후'일 때 시간 변환
        if am_pm == '오후' and hour != 12:
            hour += 12
        # '오전'일 때 시간 변환
        elif am_pm == '오전' and hour == 12:
            hour = 0
        
        return hour, minute
    except ValueError as e:
        # 변환 중 에러가 발생하면 에러 메시지를 출력하고 에러를 다시 발생시킴
        print(f"Error converting time: {e}")
        print(f"am_pm: {am_pm}, time: {time}")
        raise e


# 날짜 문자열에서 일(day)과 시간(time)을 추출하는 함수
def extract_day_and_time(date_str):
    try:
        # 날짜에서 년, 월, 일 부분 추출
        date_match = re.search(r'\d{4}년 \d{1,2}월 \d{1,2}일', date_str)
        if not date_match:
            raise ValueError("Date format incorrect")

        date_part = date_match.group()

        # 시간 부분 추출 (오전/오후 포함)
        time_match = re.search(r'(오전|오후) (\d{1,2}:\d{2})', date_str)
        if not time_match:
            raise ValueError("Time format incorrect")

        am_pm = time_match.group(1)
        time = time_match.group(2)

        return date_part, am_pm, time
    except Exception as e:
        #print(f"Error extracting day and time: {e}")
        #print(f"date_str: {date_str}")
        return None, None, None

    
# 대화 목록을 날짜별로 그룹화하는 함수
def group_messages_by_date(dialogues):
    grouped_messages = defaultdict(list)

    for message in dialogues:
        if isinstance(message, tuple):
            user, date_time, msg = message   # 튜플을 사용자, 날짜시간, 메시지로 분리
            date_str = date_time.split(' ')[1]  # 'YYYY년 MM월 DD일' 부분 추출
            grouped_messages[date_str].append(message)  # 추출한 날짜를 키로 하여 메시지를 리스트에 추가

    return grouped_messages


#누가 누구를 더 좋아한다
def compare_scores(individual_scores):
    # 사용자 이름을 추출
    users = list(individual_scores.keys())
    
    # 각 사용자의 점수를 가져오기
    user1 = users[0]
    user2 = users[1]
    score1 = individual_scores[user1]
    score2 = individual_scores[user2]
    
    # 점수 비교 후 결과 반환
    if score1 > score2:
        return f"{user1}님이 {user2}님을 더 친구라고 생각합니다."
    elif score2 > score1:
        return f"{user2}님이 {user1}님을 더 친구라고 생각합니다."
    else:
        return f"{user1}님과 {user2}님은 같은 우정도를 가지고 있습니다."


#나레이션 멘트 날리기
def narration_emotion_changes(score_list):
    result = []
    
    for person, scores in score_list.items():
        rise_detected = False
        fall_detected = False
        
        for i in range(0, len(scores)-4, 5):
            first_score = scores[i]
            last_score = scores[i + 4]

            change = last_score - first_score
            
            if change >= 5 and not rise_detected:
                result.append(f"아~ 지금 '{person}'의 우정 수치가 상승하고 있어요!")
                rise_detected = True
            elif change <= -5 and not fall_detected:
                result.append(f"아~ 지금 '{person}'의 우정 수치가 떨어지고 있어요!")
                fall_detected = True
        
        if not rise_detected and not fall_detected:
            result.append("아~주 무난한 친구들 간의 대화네요")

    return result


#기준1 - 누가 총을 대신 맞아줄 것인가?
def rule1(data, emotions_of_interest):

    result = {}
    
    # Step 1: 각 사용자별로 고마운과 설레는 값을 더한다.
    totals = {}
    for user, emotions in data.items():
        total_interest = sum(emotions.get(emotion, 0) for emotion in emotions_of_interest)
        totals[user] = total_interest

    # Step 2: 각 사용자별로 백분율을 계산한다.
    total_sum = sum(totals.values())

    for user, total_interest in totals.items():
        percentage = (total_interest / total_sum) * 100
        # 반올림하여 정수로 변환
        rounded_percentage = round(percentage)
        result[user] = rounded_percentage

    # Step 3: 최종 결과값에 따른 멘트 결정
    final_result = {}
    for user, score in result.items():
        if 0 <= score < 50:
            comment = "내가 방패가 된다."
        elif 50 <= score < 70:
            comment = "확률은 5:5 게임을 시작하지"
        elif 70 <= score <= 100:
            comment = "방패 그 자체"
        else:
            comment = "유효하지 않은 점수입니다."  # 예외 처리 (이 경우에는 없을 것 같지만)
        
        final_result[user] = (score, comment)

    return final_result


#기준2 - 흔들리지 않는 편안한 침대 같은 사람
def rule2(data, emotions_of_interest):

    result = {}
    
    # Step 1: 각 사용자별로 고마운과 설레는 값을 더한다.
    totals = {}
    for user, emotions in data.items():
        total_interest = sum(emotions.get(emotion, 0) for emotion in emotions_of_interest)
        totals[user] = total_interest

    # Step 2: 각 사용자별로 백분율을 계산한다.
    total_sum = sum(totals.values())
    for user, total_interest in totals.items():
        percentage = (total_interest / total_sum) * 100
        # 반올림하여 정수로 변환
        rounded_percentage = round(percentage)
        result[user] = rounded_percentage

    # Step 3: 최종 결과값에 따른 멘트 결정
    final_result = {}
    for user, score in result.items():
        if 0 <= score < 50:
            comment = "바닥같은 친구"
        elif 50 <= score < 70:
            comment = "장수 돌침대 별이 5개"
        elif 70 <= score <= 100:
            comment = "시몬스 침대"
        else:
            comment = "유효하지 않은 점수입니다."  # 예외 처리 (이 경우에는 없을 것 같지만)
        
        final_result[user] = (score, comment)
        

    return final_result


#기준3 - 뒷 통수 칠 사람
def rule3(data, emotions_of_interest):

    result = {}
    
    # Step 1: 각 사용자별로 고마운과 설레는 값을 더한다.
    totals = {}
    for user, emotions in data.items():
        total_interest = sum(emotions.get(emotion, 0) for emotion in emotions_of_interest)
        totals[user] = total_interest

    # Step 2: 각 사용자별로 백분율을 계산한다.
    total_sum = sum(totals.values())
    for user, total_interest in totals.items():
        percentage = (total_interest / total_sum) * 100
        # 반올림하여 정수로 변환
        rounded_percentage = round(percentage)
        result[user] = rounded_percentage

    # Step 3: 최종 결과값에 따른 멘트 결정
    final_result = {}
    for user, score in result.items():
        if 0 <= score < 50:
            comment = "신뢰 그 자체"
        elif 50 <= score < 70:
            comment = "배신을 할까? 말까?"
        elif 70 <= score <= 100:
            comment = "이미 당신 머리는 뚫려있다"
        else:
            comment = "유효하지 않은 점수입니다."  # 예외 처리 (이 경우에는 없을 것 같지만)
        
        final_result[user] = (score, comment)
        

    return final_result

