import pickle
from flask import Flask,request,app,jsonify,render_template,url_for
from model import predict_sentiment

app = Flask(__name__)

## Load the model
model = pickle.load(open('sentiment_model.pkl','rb'))

## Load the tokenizer
tokenizer = pickle.load(open('sentiment_token.pkl','rb'))

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        statement = request.form['statement']
        percent, sentiment = predict_sentiment(statement,model,tokenizer)
        percent = round((percent*100),2)
        return render_template('home.html', sentiment=sentiment, statement=statement, percent=percent)

    return render_template('home.html')

if __name__ == '__main__':
    app.run(debug=True)

