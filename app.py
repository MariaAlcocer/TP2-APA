from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

# Cargar el modelo SVD guardado
with open("svd_model.pkl", 'rb') as file:
    svd_loaded = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_id = request.form['user_id']
    movie_id = request.form['movie_id']
    prediction = svd_loaded.predict(uid=user_id, iid=movie_id)
    return jsonify({
        'user_id': user_id,
        'movie_id': movie_id,
        'prediction': prediction.est
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000) 
