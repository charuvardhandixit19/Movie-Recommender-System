import streamlit as st
import pickle
import pandas as pd

def recommend(movie):
    index = movies[movies['title'] == movie].index[0] 
    distances = similarity[index]
    movies_list = sorted(list(enumerate(distances)), key=lambda x: x[1], reverse=True)[1:6]
    recommended_list=[]
    for i in movies_list:
        recommended_list.append(movies.iloc[i[0]].title)
    return recommended_list

similarity=pickle.load(open('similarity.pkl','rb'))
movies_list=pickle.load(open('movies.pkl','rb'))
movies=pd.DataFrame(movies_list)

st.title('Movie Recommender System')

selected_movie = st.selectbox(
    "Recommend movie related to the given movie",
    movies['title'].values
)

if st.button("Recommend"):
    recommendations = recommend(selected_movie)
    for i in recommendations:
        st.write(i) 


# API = 2c3055479aa85952811b47e7e4bbcb83 