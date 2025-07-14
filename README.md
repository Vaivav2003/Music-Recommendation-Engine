# Music-Recommendation-Engine

# 🎵 Music Recommendation Engine

A machine learning-based recommendation engine that suggests songs based on user preferences and song features like tempo, genre, and popularity.

![GitHub repo size](https://img.shields.io/github/repo-size/Vaivav2003/Music-Recommendation-Engine)
![GitHub last commit](https://img.shields.io/github/last-commit/Vaivav2003/Music-Recommendation-Engine)
![GitHub license](https://img.shields.io/github/license/Vaivav2003/Music-Recommendation-Engine)

## 🚀 Overview

This project implements a content-based music recommendation system using Python. It leverages a dataset of songs with various attributes and recommends similar tracks based on user input.

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit (for optional UI)
- Jupyter Notebook

## 📁 Project Structure
📦Music-Recommendation-Engine
┣ 📜music_recommendation.ipynb
┣ 📜data.csv
┣ 📜requirements.txt
┗ 📜README.md




---

## 🔍 How It Works

1. The dataset is preprocessed to handle missing and irrelevant values.
2. Feature vectors are created using song attributes like danceability, energy, tempo, etc.
3. Cosine similarity is calculated between songs.
4. Based on a user’s input song, similar songs are recommended.

---

## ▶️ Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/Vaivav2003/Music-Recommendation-Engine.git
cd Music-Recommendation-Engine

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the Jupyter notebook or Python script

