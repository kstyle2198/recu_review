import streamlit as st
import pandas as pd
import numpy as np
import pickle
import datetime


def ranging_age(age):
    if age < 28:
        return "28미만"
    elif age >= 28 and age < 30:
        return "28to30"
    elif age >= 30 and age < 32:
        return "30to32"
    else:
        return "33이상"
    
def ranging_score(score):
    if score < 77:
        return "77미만"
    elif score >= 77 and score < 83:
        return "77to83"
    else:
        return "83이상"
    
def study_period(period):
    if period <= 6.5:
        return "6.5이하"
    else:
        return "6.5초과"

def empty_period(period):
    if period <= 2:
        return "2이하"
    else:
        return "2초과"


def main(df, 채용기준년도, selected_model):
    
    univ_grade = pd.read_excel("univ_grade.xlsx")
    merge_df = pd.merge(left=df, right=univ_grade, how='left', on='학교명')
    
    # 널값있는 행 날리기
    merge_df = merge_df.dropna()
    
    # 100점 만점 환산학점 구하기 (대학별로 학점 만점 점수 차이가 있음)
    merge_df["환산학점"] = np.round(merge_df["취득학점"].astype(float) / merge_df["만점기준"].astype(float) * 100, 1)
    
    # 재학기간 구하기
    merge_df["입학년도"] = np.int64(pd.to_datetime(merge_df["학부개시"]).dt.year)
    merge_df["졸업년도"] = np.int64(pd.to_datetime(merge_df["학부종료"]).dt.year)
    merge_df["재학기간"] = merge_df["졸업년도"] - merge_df["입학년도"]
    merge_df["공백기간"] = 채용기준년도 - merge_df["졸업년도"]
    
    # 학교등급을 점수로 변환(grade1)
    grade_dict = {"A": 1, "B": 2, "C":3, "D":4, "E":5, "F":6, "G": 7}
    merge_df = merge_df.replace({"grade1" : grade_dict})
    
    # 학교등급을 점수로 변환(grade2)
    merge_df = merge_df.replace({"grade2" : grade_dict})
    
    # 성별을 숫자로 변환
    gender_num = {"남": 0, "여": 1}
    merge_df = merge_df.replace({"sex" : gender_num})
    
    # id 칼럼을 인덱스로 변환
    # merge_df = merge_df.set_index('id')
    
    # 미사용 칼럼 삭제
    del_cols = ['name', '취득학점', '만점기준', "학부개시", "학부종료", "입학년도", "졸업년도", "check"]
    merge_df.drop(del_cols, axis=1, inplace=True)
    merge_df.head(2)
    
    st.markdown("**1단계 전처리후 데이터**")
    merge_df
    
    수정전공계열 = {
        "건축":"이공기타",
        "국문":"인문기타",
        "금속":"금속",
        "기계":"기계",
        "기타":"기타",
        "법학":"사회기타",
        "산업":"산업",
        "상경":"상경",
        "서반어":"어문기타",
        "신방":"사회기타",
        "어문":"어문기타",
        "영어":"영어",
        "이공기타":"이공기타",
        "인문기타":"인문기타",
        "재료":"재료",
        "전기전자":"전기전자",
        "전산":"전산",
        "조선해양":"조선해양",
        "중국어":"중국어",
        "토목":"이공기타",
        "화공":"화공기타",
        "안전":"안전환경",
        "환경":"안전환경"
        }
    
    수정졸업 = {
        "졸업":"졸업",
        "졸예":"졸예",
        "수료":"기타",
        "중퇴":"기타"}
    
    df = merge_df.replace({"학부전공계열" : 수정전공계열})
    df = df.replace({"졸업상태" : 수정졸업})
    
    # 합격불학격 숫자형 자료로 변환
    dict1 = {"pass": 1, "fail": 0}
    df = df.replace({"result" : dict1})
    
    df["age_range"] = df["age"].apply(ranging_age)
    df["score_range"] = df["환산학점"].apply(ranging_score)
    df["study_period"] = df["재학기간"].apply(study_period)
    df["empty_period"] = df["공백기간"].apply(empty_period)
    
    # 미사용 칼럼 삭제 
    del_cols = ["age", "학교명", "학부지역", "최종학력", "학부전공", "환산학점", "재학기간", "공백기간", "grade1"]
    df.drop(del_cols, axis=1, inplace=True)
    
    st.markdown("**2단계 전처리후 데이터 - grade2 사용**")
    df   

    # Train 당시와 같이 원핫인코딩 데이터 형태로 변환
    data = {'encoder__sex_1': 0.0,
            'encoder__입학상태_편입': 0.0,
            'encoder__졸업상태_졸업': 0.0,
            'encoder__졸업상태_졸예': 0.0,
            'encoder__학부전공계열_기계': 0.0,
            'encoder__학부전공계열_기타': 0.0,
            'encoder__학부전공계열_사회기타': 0.0,
            'encoder__학부전공계열_산업': 0.0,
            'encoder__학부전공계열_상경': 0.0,
            'encoder__학부전공계열_안전환경': 0.0,
            'encoder__학부전공계열_어문기타': 0.0,
            'encoder__학부전공계열_영어': 0.0,
            'encoder__학부전공계열_이공기타': 0.0,
            'encoder__학부전공계열_인문기타': 0.0,
            'encoder__학부전공계열_재료': 0.0,
            'encoder__학부전공계열_전기전자': 0.0,
            'encoder__학부전공계열_전산': 0.0,
            'encoder__학부전공계열_조선해양': 0.0,
            'encoder__학부전공계열_중국어': 0.0,
            'encoder__학부전공계열_화공기타': 0.0,
            'encoder__grade2_2': 0.0,
            'encoder__grade2_3': 0.0,
            'encoder__grade2_4': 0.0,
            'encoder__grade2_5': 0.0,
            'encoder__grade2_6': 0.0,
            'encoder__grade2_7': 0.0,
            'encoder__age_range_28미만': 0.0,
            'encoder__age_range_30to32': 0.0,
            'encoder__age_range_33이상': 0.0,
            'encoder__score_range_77미만': 0.0,
            'encoder__score_range_83이상': 0.0,
            'encoder__study_period_6.5초과': 0.0,
            'encoder__empty_period_2초과': 0.0}
    
    
    if df['sex'].values[0] == "남성":
        data['encoder__sex_1'] = 1
        
    if df['입학상태'].values[0] == "편입":
        data['encoder__입학상태_편입'] = 1 
        
    if df['졸업상태'].values[0] == "졸업":
        data['encoder__졸업상태_졸업'] = 1  
    elif df['졸업상태'].values[0] == "졸예":
        data['encoder__졸업상태_졸예'] = 1  
        
    if df['학부전공계열'].values[0] == "기계":
        data['encoder__학부전공계열_기계'] = 1  
    elif df['학부전공계열'].values[0] == "기타":
        data['encoder__학부전공계열_기타'] = 1  
    elif df['학부전공계열'].values[0] == "사회기타":
        data['encoder__학부전공계열_사회기타'] = 1  
    elif df['학부전공계열'].values[0] == "산업":
        data['encoder__학부전공계열_산업'] = 1  
    elif df['학부전공계열'].values[0] == "상경":
        data['encoder__학부전공계열_상경'] = 1  
    elif df['학부전공계열'].values[0] == "안전환경":
        data['encoder__학부전공계열_안전환경'] = 1  
    elif df['학부전공계열'].values[0] == "어문기타":
        data['encoder__학부전공계열_어문기타'] = 1  
    elif df['학부전공계열'].values[0] == "영어":
        data['encoder__학부전공계열_영어'] = 1 
    elif df['학부전공계열'].values[0] == "이공기타":
        data['encoder__학부전공계열_이공기타'] = 1  
    elif df['학부전공계열'].values[0] == "인문기타":
        data['encoder__학부전공계열_인문기타'] = 1  
    elif df['학부전공계열'].values[0] == "재료":
        data['encoder__학부전공계열_재료'] = 1  
    elif df['학부전공계열'].values[0] == "전기전자":
        data['encoder__학부전공계열_전기전자'] = 1  
    elif df['학부전공계열'].values[0] == "전산":
        data['encoder__학부전공계열_전산'] = 1  
    elif df['학부전공계열'].values[0] == "조선해양":
        data['encoder__학부전공계열_조선해양'] = 1      
    elif df['학부전공계열'].values[0] == "중국어":
        data['encoder__학부전공계열_중국어'] = 1      
    elif df['학부전공계열'].values[0] == "화공기타":
        data['encoder__학부전공계열_화공기타'] = 1  
        
    if df['grade2'].values[0] == 2:
        data['encoder__grade2_2'] = 1 
    elif df['grade2'].values[0] == 3:
        data['encoder__grade2_3'] = 1 
    elif df['grade2'].values[0] == 4:
        data['encoder__grade2_4'] = 1 
    elif df['grade2'].values[0] == 5:
        data['encoder__grade2_5'] = 1 
    elif df['grade2'].values[0] == 6:
        data['encoder__grade2_6'] = 1 
    elif df['grade2'].values[0] == 7:
        data['encoder__grade2_7'] = 1 
        
    if df['age_range'].values[0] == "28미만":
        data['encoder__age_range_28미만'] = 1 
    elif df['age_range'].values[0] == "30to32":
        data['encoder__age_range_30to32'] = 1 
    elif df['age_range'].values[0] == "33이상":
        data['encoder__age_range_33이상'] = 1         
        
    if df['score_range'].values[0] == "77미만":
        data['encoder__score_range_77미만'] = 1 
    elif df['score_range'].values[0] == "83이상":
        data['encoder__score_range_83이상'] = 1 

    if df['study_period'].values[0] == "6.5초과":
        data['encoder__study_period_6.5초과'] = 1 

    if df['empty_period'].values[0] == "2초과":
        data['encoder__empty_period_2초과'] = 1         
        
    df1 = pd.DataFrame.from_dict(data, orient='index').T
    
    st.markdown("**원핫인코딩(OneHotEncoding)**")
    df1
    
    
    # train시 onehot encoding 단계에서 (컬럼이 포함된) df를 받아 ColumnTransformer를 적용후 모델학습을 시켰었기 때문에,
    # 예측시에도 칼럼이 포함된 df형태로 데이터를 모델에 던져줘야 함 (여기서는 df1 / 넘파이 어레이 형태 X)
    
    if selected_model == "LogisticRegression":
    
        with open('./models/model(logistic_regression).sav','rb') as pickle_filename:
            best_model = pickle.load(pickle_filename)
        
        predicted = best_model.predict(df1)
        score = best_model.predict_proba(df1)        

        col01, col02, col03 = st.columns([1, 1, 8])
        with col01: 
            st.markdown(":green[**Hard Label(0: fail, 1: pass)**]")  
            predicted
        with col02:
            st.markdown(":green[**Soft Label(확률)**]")
            score
            
    elif selected_model == "RandomForest":
        with open('./models/model(random_forest).sav','rb') as pickle_filename:
            best_model = pickle.load(pickle_filename)
        
        predicted = best_model.predict(df1)
        score = best_model.predict_proba(df1)
        
        col001, col002, col003 = st.columns([1, 1, 8])
        with col001: 
            st.markdown(":green[**Hard Label(0: fail, 1: pass)**]")  
            predicted
        with col002:
            st.markdown(":green[**Soft Label(확률)**]")
            score

    elif selected_model == "GaussianNB":
        with open('./models/model(gaussian_nb).sav','rb') as pickle_filename:
            best_model = pickle.load(pickle_filename)
        
        predicted = best_model.predict(df1)
        score = best_model.predict_proba(df1)
        
        col001, col002, col003 = st.columns([1, 1, 8])
        with col001: 
            st.markdown(":green[**Hard Label(0: fail, 1: pass)**]")  
            predicted
        with col002:
            st.markdown(":green[**Soft Label(확률)**]")
            score    

if __name__ == "__main__":
    
    st.set_page_config(layout="wide", page_title="AI_Copilot[HR]")
    st.markdown("#### :red[AI Copilot] Series - :blue[HR/Recruit]🍀")
    st.markdown("### :red[채용 학습모델] 기반 :blue[서류전형 합격률 예측 플랫폼]")
    # st.markdown("---")
    
    ####################
    학부전공 = pd.read_csv("학부전공.csv")
    학부전공 = 학부전공['학부전공'].tolist()
    
    with st.expander("🎈 :blue[**지원자 조건 설정**] - 미사용 조건은 비활성화 상태임"):
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        with col1:
            name = st.selectbox("성명", ["박보검"], disabled=True)
            연령후보군 = [27, 29, 31, 33]
            default_연령 = 연령후보군.index(29)
            age = st.selectbox("연령", 연령후보군, index=default_연령)
            sex = st.selectbox("성별", ["남성", "여성"])
            
        with col2:
            입학상태 = st.selectbox("입학상태", ["입학", "편입"])
            졸업상태 = st.selectbox("졸업상태", ["졸업", "졸예"])   # "수료", "중퇴"는 대학원급에서 발생하여 제외
        
        with col3:
            학교명 = st.selectbox("학교명", ["서울대", "고려대", "한양대", "동국대", "경북대", "울산대", "전남대", "계명대"])
            학부지역 = st.selectbox("학부지역", ["서울", "경기"], disabled=True)   # 현재 미사용

            
        with col4:
            학부전공계열 = st.selectbox("학부전공계열", ["기계", "전기전자", "산업", "조선해양", "이공기타", "전산", "상경", "영어", "국문", "금속", "건축", "기타","법학", "서반아어", "신방", "어문", "인문기타", "재료", "중국어", "토목", "화공", "안전"])
            학부전공 = st.selectbox("학부전공", 학부전공, disabled=True)           # 현재 미사용
            최종학력 = st.selectbox("최종학력", ["학사", "석사"], disabled=True)   # 현재 미사용
       
        with col5:
            학부개시 = st.date_input("학부개시", datetime.date(2019, 3, 1))
            학부종료 = st.date_input("학부종료", datetime.date(2023, 2, 1))
            
        with col6:
            학점후보군 = [2.8, 3.0, 3.5, 3.75, 4.0, 4.3]
            default_학점 = 학점후보군.index(3.5)
            취득학점 = st.selectbox("취득학점", 학점후보군, index=default_학점)
            만점기준 = st.selectbox("만점기준", [4.5, 4.0])
            model_select = st.selectbox("✔️ :red[**모델 선택**]", ["LogisticRegression", "RandomForest", "GaussianNB"])  #트레인시 로지스틱이 1등, 가우시안엔비가 꼴찌
        
    data = {"name": name, "age": age, "sex": sex, "학교명": 학교명, "학부지역": 학부지역, "입학상태": 입학상태, "졸업상태": 졸업상태, "학부전공계열": 학부전공계열, "학부전공": 학부전공, 
            "최종학력": 최종학력, "학부개시": 학부개시, "학부종료": 학부종료, "취득학점": 취득학점, "만점기준": 만점기준}
    df = pd.DataFrame.from_dict(data, orient='index')
    df = df.T
    
    # st.markdown("인풋 데이터")
    df

    ##########################
    main(df, 2023, model_select)




