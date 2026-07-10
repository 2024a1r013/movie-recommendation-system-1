import streamlit as st
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set up the page title
st.set_page_config(page_title="Movie Recommender", page_icon="🎬")
st.title("🎬 Movie Recommendation System")
st.write("Enter a movie you love, and we'll recommend similar movies based on genres, cast, and directors!")

# Cache the data loading so the app runs fast
@st.cache_data
def load_data_and_train():
    # Load dataset
    movies_data = pd.read_csv('movies.csv') 
    
    # Select features and handle missing values
    selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']
    for feature in selected_features:
        movies_data[feature] = movies_data[feature].fillna('')
        
    # Combine features and calculate similarity
    combined_features = movies_data['genres']+' '+movies_data['keywords']+' '+movies_data['tagline']+' '+movies_data['cast']+' '+movies_data['director']
    vectorizer = TfidfVectorizer()
    feature_vectors = vectorizer.fit_transform(combined_features)
    similarity = cosine_similarity(feature_vectors)
    
    return movies_data, similarity

# Attempt to load data
try:
    movies_data, similarity = load_data_and_train()
    list_of_all_titles = movies_data['title'].tolist()

    # Create the search bar
    movie_name = st.text_input("Enter your favorite movie name:")

    # Create the search button
    if st.button("Get Recommendations"):
        if movie_name:
            find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)
            
            if find_close_match:
                close_match = find_close_match[0]
                # Get the index
                index_of_the_movie = movies_data[movies_data.title == close_match].index[0]
                similarity_score = list(enumerate(similarity[index_of_the_movie]))
                sorted_similar_movies = sorted(similarity_score, key = lambda x:x[1], reverse = True)
                
                st.success(f"Because you liked **{close_match}**, we recommend:")
                
                # Display top 10 movies (skipping the first one, which is the searched movie itself)
                for i, movie in enumerate(sorted_similar_movies[1:11], 1):
                    index = movie[0]
                    title_from_index = movies_data.iloc[index]['title']
                    st.write(f"**{i}.** {title_from_index}")
            else:
                st.error("No close matches found. Please check your spelling or try another movie.")
        else:
            st.warning("Please enter a movie name first.")

except FileNotFoundError:
    st.error("Dataset 'movies.csv' not found. Please make sure it is in the same folder as this app.")