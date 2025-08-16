from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

# Load the trained model and vectorizer
model = joblib.load('model.pkl')
vec = joblib.load('vectorizer.pkl')

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = ''
    user_input = ''
    if request.method == 'POST':
        user_input = request.form['headline']
        user_vect = vec.transform([user_input])
        pred = model.predict(user_vect)
        prediction = 'Clickbait' if pred[0] == 1 else 'Not Clickbait'
    return render_template('index.html', prediction=prediction, user_input=user_input)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)
