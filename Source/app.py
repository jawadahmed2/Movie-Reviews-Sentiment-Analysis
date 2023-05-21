# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


# Load the logistic Forest CLassifier model
print('Wait Model Is Loading')
filename = 'LR_model.pickle'
classifier = pickle.load(open(filename, 'rb'))
data = pd.read_csv('train_review_data.csv')

X = data['review']

tfidf = TfidfVectorizer()

X = tfidf.fit_transform(X)

print('Successfully Loaded')

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
	if request.method != 'POST':
		return
	review_text = str(request.form['review'])



	data = tfidf.transform([review_text])

	my_prediction = (classifier.predict(data))[0]

	return render_template('index.html', prediction=my_prediction.upper())

if __name__ == '__main__':
	app.run(debug=True)