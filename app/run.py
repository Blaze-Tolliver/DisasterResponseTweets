import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Histogram
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    """Function to tokenize, lemmatize, and case normalize a tweet.

    Args:
        text (str): tweet text.
    Returns:
        clean_tokens(list): list of cleaned tokens from the tweet.

    """
    
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('Categorized_Tweets', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    """Function to create plotly graphs, encode them in JSON, and render them to master.html using Flask.

    Args:
        None.
    Returns:
        render_template(object): Passes the graphs to the master.html page.

    """
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    category_counts = df.drop(['id','message','original', 'genre'], axis = 1).sum(axis = 1, numeric_only = True, skipna = True)
    
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Histogram(
                    x=category_counts
                    
                )
            ],

            'layout': {
                'title': 'Histogram of Categories per Tweet in Original Dataset',
                'yaxis': {
                    'title': "Tweet Count"
                },
                'xaxis': {
                    'title': "Number of Categories Matched"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    """Function to take user query and run the classification model to predict the categorie(s)
    that the query matches.

    Args:
        None.
    Returns:
        render_template(object): Passes the classification results and query text to 'go.html' page.

    """
    
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    """Function to run the Flask app.

    Args:
        None.
    Returns:
        None.

    """
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()