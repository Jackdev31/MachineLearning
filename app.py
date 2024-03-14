from flask import Flask, render_template, request
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import random

app = Flask(__name__)

data = pd.read_csv('imgsong.csv')
data = data.dropna()

X = data.drop(columns=["song_title", "img_song"])
y = data["song_title"]

model = DecisionTreeClassifier()
model.fit(X.values, y)

genre_label = {
    1: "Pop", 2: "Rock", 3: "Hip-Hop", 4: "Metal", 5: "R&B"
}
preferred = {
    1: "International", 2: "Local"
}

@app.route('/')
def index():
    return render_template('index.html', genre_label=genre_label, preferred=preferred)

@app.route('/suggested', methods=['POST'])
def predict():
    song_genre = int(request.form['song_genre'])
    prefer = int(request.form['preferred']) 

    predict_song = model.predict([[song_genre, prefer]])[0] 

    filtered_data = data[(data['song_genre'] == song_genre) & (data['preferred'] == prefer)]

    recommended_songs = random.sample(filtered_data[['song_title', 'img_song']].values.tolist(), 5)

    for song in recommended_songs:
        song[1] = f"img/{song[1]}"

    return render_template('index.html', predict_song=predict_song,
        recommended_songs=recommended_songs,
        genre_label=genre_label,
        preferred=preferred)

if __name__ == '__main__':
    app.run(debug=True, port=8080)
