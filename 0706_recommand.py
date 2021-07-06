from typing import Counter
import pandas as pd
import numpy as np
import warnings; warnings.filterwarnings('ignore')
import os

os.chdir('/Users/heechankang/projects/pythonworkspace/git_study/nlp')

movies = pd.read_csv('tmdb_5000_movies.csv')

print(movies.shape)

movies.head(1)

movies_df= movies[['id','title','genres','vote_average','vote_count','popularity','keywords','overview']]
movies_df

type(movies_df['genres'][0])
print(movies_df['genres'][0])
# 이게 통째로 str임.. 이대로는 활용 불가

from ast import literal_eval # 문자열 파싱 라이브러리
movies_df['genres'] = movies_df['genres'].apply(literal_eval)
movies_df['keywords'] = movies_df['keywords'].apply(literal_eval)

movies_df['genres'][0][0]
movies_df['keywords']
type(movies_df['genres'][0])
# 리스트로 변경

movies_df['genres']

# key 값에 따른 value 값만 추출
movies_df['genres'] = movies_df['genres'].apply(lambda x: [y['name'] for y in x])
movies_df['keywords'] = movies_df['keywords'].apply(lambda x: [y['name'] for y in x])

movies_df[['genres', 'keywords']]

##### 전처리 끝

##### 장르 컨텐츠 유사도 측정

# 할 것
# 1. 문서를 토큰 리스트로 변환
# 2. 각 문서에서 토큰의 출현빈도를 확인
# 3. 각 문서를 BOW(Bag of Words) 인코딩 벡터로 변환

from sklearn.feature_extraction.text import CountVectorizer
corpus = [
    'This is the first document.',
    'This is the second second document.',
    'And the third one.',
    'Is this the first document?',
    'The last document?',
]

vect = CountVectorizer()
vect.fit(corpus)
vect.vocabulary_

movies_df['genres']

myTuple = 'John', 'Peter', 'Vicky'

x = '-'.join(myTuple)
print(x)


# CountVectorizer 를 적용하기 위해, 공백문자로 word 구분해주기
movies_df['genres_literal'] = movies_df['genres'].apply(lambda x: (' ').join(x))
movies_df['genres_literal']

count_vect = CountVectorizer(min_df = 0, ngram_range = (1, 2))
count_vect.vocabulary_
# min_df: 최소 빈도, ngram_range: 단어의 묶음 최소, 최대 수.
genre_mat = count_vect.fit_transform(movies_df['genres_literal'])
genre_mat.shape
print(genre_mat)
# CountVectorizer로 학습시켰더니 총 4803개 영화에 대해 276개의 장르 매트릭스 형성됨.

count_vect2 = CountVectorizer(min_df = 1, ngram_range=(1,1))
genre_mat2 = count_vect2.fit_transform(movies_df['genres_literal'])
genre_mat2.shape
print(genre_mat2)
# min_df = 1인 경우, 총 4803개 영화에 대해 22개의 장르 매트릭스가 형성됨.


# 코사인 유사도 분석
from sklearn.metrics.pairwise import cosine_similarity
genre_sim = cosine_similarity(genre_mat, genre_mat)
print(genre_sim.shape)
print(genre_sim[:5])
print(genre_sim)


# agrsort 개념 설명 - 인덱스를 뽑아줌
x = np.array([3,1,2,])
np.argsort(x)

x = np.array([[0,3],[2,2]])
np.argsort(x)

ind = np.argsort(x, axis = 0)
ind


# 
genre_sim_sorted_ind = genre_sim.argsort()[:, ::-1] # 유사도 순위에 따른 인덱스를 오름차순으로 정렬
genre_sim_sorted_ind

# 'title' 이 'The Godfather'인 행 추출
title_movie = movies_df[movies_df['title'] == 'The Godfather']
title_movie

title_index = title_movie.index.values

similar_indexes = genre_sim_sorted_ind[title_index, :10]
similar_indexes

similar_indexes = similar_indexes.reshape(-1)
similar_indexes

movies_df.iloc[similar_indexes]


def find_sim_movies_verl(df, sorted_ind, title_name, top_n = 10):
    title_movie = df[df['title']==title_name]

    title_index = title_movie.index.values
    similar_indexes = sorted_ind[title_index, :(top_n)]

    print(similar_indexes)

    similar_indexes = similar_indexes.reshape(-1)

    return df.iloc[similar_indexes]


#############################################################################
#############################################################################
#############################################################################

movies_df['genres_literal'] = movies_df['genres'].apply(lambda x : (' ').join(x))
movies_df['genres_literal']

# CountVectorizer로 학습, 4803개 영화의 276개 장르 매트릭스 생성
count_vect = CountVectorizer(min_df = 0, ngram_range=(1,2))
# min_df: 단어장에 들어갈 최소 빈도, ngram_range: 1 <= n <= 2
genre_mat = count_vect.fit_transform(movies_df['genres_literal'])
print(genre_mat.shape)

print(genre_mat)

# CountVectorizer로 학습시켰더니 4803개 영화에 대한 22개 장르의 '장르 매트릭스'가 생성되었다.
count_vect2 = CountVectorizer(min_df=1, ngram_range=(1,1)) 
#min_df: 단어장에 들어갈 최소빈도, ngram_range: 1 <= n <= 2
genre_mat2 = count_vect2.fit_transform(movies_df['genres_literal'])
print(genre_mat2.shape)

print(genre_mat2)

# CountVectorizer로 학습시켰더니 4803개 영화에 대한 276개 장르의 '장르 매트릭스'가 생성되었다.
count_vect2 = CountVectorizer(min_df=0, ngram_range=(1,1)) #min_df: 단어장에 들어갈 최소빈도, ngram_range: 1 <= n <= 2
genre_mat2 = count_vect2.fit_transform(movies_df['genres_literal'])
print(genre_mat2.shape)

# 코사인 유사도에 의해 4803개 영화 각각 유사한 영화들이 계산됨
from sklearn.metrics.pairwise import cosine_similarity
genre_sim = cosine_similarity(genre_mat, genre_mat)

print(genre_sim.shape)
print(genre_sim[:5])
print(genre_sim)

# 1차원 배열일 때 
x = np.array([3, 1, 2])
np.argsort(x)

# 2차원 배열 
x = np.array([[0, 3],[2, 2]])
x

ind = np.argsort(x, axis=0)  # sorts along first axis (down)
ind

# 자료를 정렬하는 것이 아니라 순서만 알고 싶다면 argsort
# 유사도가 높은 영화를 앞에서부터 순서대로 보여줌
# 0번째 영화의 경우 유사도 순서 : 0번, 3494번, 813번, ..., 2401 순서
genre_sim_sorted_ind = genre_sim.argsort()[:, ::-1] # 전체를 -1칸 간격으로
print(genre_sim_sorted_ind[:5])

genre_sim_sorted_ind

# 인자로 입력된 movies_df DataFrame에서 'title' 컬럼이 입력된 title_name 값인 DataFrame추출
title_movie = movies_df[movies_df['title'] == 'The Godfather']
title_movie

# title_named을 가진 DataFrame의 index 객체를 ndarray로 반환하고 
# sorted_ind 인자로 입력된 genre_sim_sorted_ind 객체에서 유사도 순으로 top_n 개의 index 추출
title_index = title_movie.index.values
title_index

similar_indexes = genre_sim_sorted_ind[title_index, :10]
similar_indexes

# 추출된 top_n index들 출력. top_n index는 2차원 데이터 임.
# dataframe에서 index로 사용하기 위해서 2차원 array를 1차원 array로 변경
similar_indexes = similar_indexes.reshape(-1)
print(similar_indexes)

movies_df.iloc[similar_indexes]

# 최초 추천함수
def find_sim_movie_ver1(df, sorted_ind, title_name, top_n=10):
    
    # 인자로 입력된 movies_df DataFrame에서 'title' 컬럼이 입력된 title_name 값인 DataFrame추출
    title_movie = df[df['title'] == title_name]
    
    # title_named을 가진 DataFrame의 index 객체를 ndarray로 반환하고 
    # sorted_ind 인자로 입력된 genre_sim_sorted_ind 객체에서 유사도 순으로 top_n 개의 index 추출
    title_index = title_movie.index.values
    similar_indexes = sorted_ind[title_index, :(top_n)]
    
    # 추출된 top_n index들 출력. top_n index는 2차원 데이터 임.
    # dataframe에서 index로 사용하기 위해서 1차원 array로 변경
    print(similar_indexes)    
    # 2차원 데이터를 1차원으로 변환
    similar_indexes = similar_indexes.reshape(-1)
    
    return df.iloc[similar_indexes]


# 갓파더와 유사한 영화 10개 추천
similar_movies = find_sim_movie_ver1(movies_df, genre_sim_sorted_ind, 'The Godfather',10)
similar_movies[['title', 'vote_average', 'genres', 'vote_count']]
# 문제 ; 평점 기반으로 추천하고자 하는데, vote_count가 낮은 영화는 제외하고 싶음

# 평점 기준으로 정렬(sort_values)해보니 평가횟수(vote_count)가 낮은 것들도 포함됨. 
# 평가횟수 반영한 평점(가중평점)이 필요함, 내림차순으로 상위 10개 추출 
movies_df[['title','vote_average','vote_count']].sort_values('vote_average', ascending=False)[:10]


# 가중평점을 반영한 영화 추천
# 상위 60%에 해당하는 vote_count를 최소 투표 횟수인 m으로 지정
C = movies_df['vote_average'].mean()
m = movies_df['vote_count'].quantile(0.6)

# C: 전체 영화에 대한 평균평점 = 약 6점
# m: 평점을 부여하기 위한 최소 투표 횟수 = 370회(상위 60% 수준)
print('C:',round(C,3), 'm:',round(m,3))


# 가중평점을 계산하는 함수
def weighted_vote_average(record):
    v = record['vote_count']
    R = record['vote_average']
    
    return ( (v/(v+m)) * R ) + ( (m/(m+v)) * C )   

# 기존 데이터에 가중평점 칼럼 추가
movies_df['weighted_vote'] = movies_df.apply(weighted_vote_average, axis=1)

movies_df.head()

# 가중평점(weighted_vote) 기준으로 영화 순위 뽑아보기(내림차순)
movies_df[['weighted_vote','title','vote_average','vote_count','genres']].sort_values('weighted_vote', ascending=False)[:10]

# 추천 ver2. 먼저 장르 유사성 높은 영화 20개 선정 후, 가중평점순 10개 선정
def find_sim_movie_ver2(df, sorted_ind, title_name, top_n=10):
    title_movie = df[df['title'] == title_name]
    title_index = title_movie.index.values
    
    # top_n의 2배에 해당하는 쟝르 유사성이 높은 index 추출
    similar_indexes = sorted_ind[title_index, :(top_n*2)]
    similar_indexes = similar_indexes.reshape(-1)

    # 기준 영화 index는 제외(기존에 본 영화는 제외하고 추천, title index가 겹치지 않는 영화 추천)
    similar_indexes = similar_indexes[similar_indexes != title_index]
    
    # top_n의 2배에 해당하는 후보군에서 weighted_vote 높은 순으로 top_n 만큼 추출 
    return df.iloc[similar_indexes].sort_values('weighted_vote', ascending=False)[:top_n]


# 갓파더에 대해 장르 유사성과 가중평점을 반영한 추천영화 10개
similar_movies = find_sim_movie_ver2(movies_df, genre_sim_sorted_ind, 'The Godfather',10)
similar_movies[['title', 'vote_average', 'weighted_vote', 'genres', 'vote_count']]


# 스파이더맨3 기준으로 다시 추천
similar_movies = find_sim_movie_ver2(movies_df, genre_sim_sorted_ind, 'Spider-Man 3',10)
similar_movies[['title', 'vote_average', 'weighted_vote', 'genres', 'vote_count']]


# 에너미앳더게이트
similar_movies = find_sim_movie_ver2(movies_df, genre_sim_sorted_ind, 'Enemy at the Gates',10)
similar_movies[['title', 'vote_average', 'weighted_vote', 'genres', 'vote_count']]