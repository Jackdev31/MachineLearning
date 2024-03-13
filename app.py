from flask import Flask, render_template, request
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import random

app = Flask(__name__)

data = pd.read_csv('testingko.csv')
data = data.dropna()
X = data.drop(columns=["songs"])
y = data["songs"]

model = DecisionTreeClassifier()
model.fit(X.values, y)

genre_label = {
    1: "Pop", 2: "Rock", 3: "Emo", 4: "K-pop", 5: "Metal"
}

artist_genre = {
    1: 1, 2: 1, 3: 1, 4: 1, 5: 1,  
    6: 2, 7: 2, 8: 2, 9: 2, 10: 2,
    11:3, 12:3, 13:3, 14:3, 15:3,
    16:4, 17:4, 18:4, 19:4, 20:4,
    21:5, 22:5, 23:5, 24:5, 25:5  
}


genre_artists = {
    1: {1: "Ed Sheeran", 2: "Taylor Swift", 3: "Justin Bieber", 4: "Ariana Grande", 5: "Katy Perry"}, #Pop
    2: {6: "Eagles", 7: "AC DC", 8: "Air Supply", 9: "The Beatles", 10: "Fallout Boy"}, #Rock
    3: {11: "Paramore", 12: "My Chemical Romance", 13: "All Time Low", 14: "Linkin Park", 15: "Yellowcard"}, #Emo
    4: {16: "IU", 17: "EXO", 18: "GOT 7", 19: "BTS", 20: "ENHYPEN"}, #K-pop
    5: {21: "Black Sabbath", 22: "Megadeth", 23: "Judas Priest", 24: "Metallica", 25: "Dream Theater"} #Metal
}

@app.route('/')
def index():
    return render_template('index.html',
        genres=genre_label,
        artists=genre_artists)

@app.route('/recommend', methods=['POST'])
def predict():
    genre = int(request.form['genre'])
    artist = int(request.form['artist'])
    
    artist_songs = data[data['artist'] == artist]['songs'].tolist()
    
    popular_random = random.sample(artist_songs, 5)
    
    selected_artist = genre_artists[genre][artist]
    
    return render_template('index.html',
        popular_random=popular_random,
        genres=genre_label,
        artists=genre_artists[genre],
        selected_genre=genre,
        artistname=selected_artist)


if __name__ == '__main__':
    app.run(debug=True, port=8080)
