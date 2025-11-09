from flask import Flask, render_template, request
import joblib

app = Flask(__name__)
model = joblib.load('fake_news_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['POST'])
def predict():
    news = request.form['news']
    prediction = model.predict([news])[0]
    prob = model.predict_proba([news])[0]
    confidence = round(max(prob) * 100, 2)
    label = 'Real' if prediction == 0 else 'Fake'
    return render_template('result.html', label=label, confidence=confidence)

if __name__ == '__main__':
    app.run(debug=True)