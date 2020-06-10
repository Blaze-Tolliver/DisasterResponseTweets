# Disaster Response Pipeline Project
1. The README file includes a summary of the project, how to run the Python scripts and web app, and an explanation of the files in the repository. 

# Libraries
1. Standard Libraries: pandas, numpy, re, pickle
2. Additional Libraries: sqlalchemy, sklearn, json, plotly, flask

# Project Motivation
1. During a disaster it would be extremely helpful if tweets to aid organizations could be automatically sorted by category so they could be routed to the appropriate agency/organization.

# Project Summary
1. This program takes categorized disaster tweets, cleans the data, splits the data into train and test datasets, trains a multi-label classification model on the training data, shows the model's precision, recall, f1-score, and support for each message category, saves the model to a pickle file, & runs a Flask web app that runs user text queries through the model to predict their response categorie(s).  
2. I ended up using a OneVsRest multi-target classifier in combination with LinearSVC as it showed marginally better results than the MultiOutput classifier with RandomForest or Multinomial Naive Bayes. I still need to invest more time in refining the model, but it runs very slowly on Udacity's IDE. I will further refine the model for my portfolio.

3. The model predicts categories that show up frequently in the training dataset much better than it predicts rare categories.

# Files
- app
	 - template
		- master.html  # main page of web app
		- go.html  # classification result page of web app
		- run.py  # Flask file that runs app

- data
	- disaster_categories.csv  # tweet category data to process 
	- disaster_messages.csv  # tweet text data to process
	- process_data.py
	- DisasterResponse.db   # database to save clean data to

- models
	- train_classifier.py
	- classifier.pkl  # saved model 

- README.md

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
