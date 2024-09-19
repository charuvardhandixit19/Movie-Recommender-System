import numpy as np 
import pandas as pd
import ast
import nltk 
import pickle

# IMPORT OS BOTH DATASET
movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')
# print(movies.shape)
# print(movies.head())

# MERGING OF BOTH DATASET
movies=movies.merge(credits,on='title')
# print(movies.merge(credits,on='title').shape)
# print(movies.head())

# REMOVING UNNECESSARY COLUMNS
movies=movies[['movie_id','title','overview','genres','keywords','cast','crew']]
# print(movies.head())
# print(movies.isnull().sum())

# DROPING ROW WITH MISSING INFORMATION
movies.dropna(inplace=True)
# print(movies.isnull().sum())
# print(movies.duplicated().sum())

#  FORMATION OF THE DATA INSIDE THE COLUMN
# print(movies.iloc[0].genres)
def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L   
movies['genres']=movies['genres'].apply(convert) 
# print(movies.iloc[0].genres)
movies['keywords']=movies['keywords'].apply(convert)
# print(movies.iloc[0].keywords)
def convertCast(obj):
    L = []
    counter =0
    for i in ast.literal_eval(obj):
        if counter != 3:
            L.append(i['name'])
            counter +=1
        else:
            break
    return L 
movies['cast']=movies['cast'].apply(convertCast)
# print(movies.iloc[0].cast)
def findDirector(obj):
    L=[]
    for i in ast.literal_eval(obj):
        if i['job']=='Director':
            L.append(i['name'])
            break   
    return L   
movies['crew']=movies['crew'].apply(findDirector)
# print(movies.iloc[0].crew)
# print(movies.head())
movies['overview']=movies['overview'].apply(lambda x:x.split())
# print(movies.iloc[0].overview)
movies['genres']=movies['genres'].apply(lambda x:[i.replace(" ","") for i in x]) 
# print(movies.iloc[0].genres)
movies['keywords']=movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x]) 
# print(movies.iloc[0].keywords)
movies['cast']=movies['cast'].apply(lambda x:[i.replace(" ","") for i in x]) 
# print(movies.iloc[0].cast)
movies['crew']=movies['crew'].apply(lambda x:[i.replace(" ","") for i in x]) 
# print(movies.iloc[0].crew)

# CREATION OF NEW COLUMN CALLED "TAG" CONCATIONATION OF OVERVIEW+GENRES+CAST+CREW+KEYWORDS.
movies['tag']=movies['overview']+movies['genres']+movies['keywords']+movies['cast']+movies['crew']

# NEW DATA FRAME
new_df=movies[['movie_id','title','tag']]
# print(new_df.head())
new_df['tag']=new_df['tag'].apply(lambda x:" ".join(x))
new_df['tag']=new_df['tag'].apply(lambda x:x.lower())
# print(new_df.head())

#------------------------------------------------------------------- VECTORIZATION  ------------------------------------------------------------
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=3000,stop_words='english')
vectors = cv.fit_transform(new_df['tag']).toarray()
# print(vectors[0])
len=cv.get_feature_names_out()
# print(len)
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
def stem(text):
    y=[]
    for i in text.split():
        y.append(stemmer.stem(i))
    return " ".join(y)

new_df['tag']=new_df['tag'].apply(stem) 

#CALCULATING THE COSINE DISTANCE BETWEEN VECTORS
from sklearn.metrics.pairwise import cosine_similarity
similarity=cosine_similarity(vectors)
# print(cosine_similarity(vectors).shape)

# MAIN FUNCTION FOR MOVIE RECOMENDATION
def recommend(movie):
    index = new_df[new_df['title'] == movie].index[0] 
    distances = similarity[index]
    movies_list = sorted(list(enumerate(distances)), key=lambda x: x[1], reverse=True)[1:6]
    for i in movies_list:
        print(new_df.iloc[i[0]].title)

# print(recommend('Avatar'))        


# Sending file for reecommendation website
pickle.dump(new_df.to_dict(),open('movies.pkl','wb')) 
pickle.dump(similarity,open('similarity.pkl','wb'))






