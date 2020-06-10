import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """Function to bring in the messages & categories csv files, convert them to pandas dfs, and merge them on column 'id'.

    Args:
        messages_filepath (str): filepath to the csv file containing the disaster tweets.
        categories_filepath (str): filepath to the csv file containing the disaster tweet categories.
    Returns:
        df(dataframe): the merged dataset of messages and categories.

    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on = 'id')
    return df
    
    
def clean_data(df):
    """Function to convert output of load_data() to a dataset ready for multi-label classification. 
    Converts categories column from string to a binary array with categories as column names. Removes the 
    rows where category 'related' = 2, because this is undefined. Removes category 'child_alone', because
    no tweets are categorized as 'child_alone'.

    Args:
        df (dataframe): output dataframe from load_data().
    Returns:
        df(dataframe): cleaned dataframe.

    """
    
    categories = df.categories.str.split(';', expand = True)
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1])
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    
    df = df.drop('categories', axis = 1)
    df = pd.concat([df, categories], axis = 1)
    df = df[~df.duplicated(keep = 'first')]
    df = df[df.related != 2]
    df = df.drop('child_alone', axis = 1)
    return df

def save_data(df, database_filename):
    """Function to save the cleaned dataframe to table 'Categorized_Tweets' an sqlite database
    specified by database_filename.

    Args:
        df (database): cleaned dataframe from clean_data().
        database_filename (str): name of the database to store the cleaned dataframe in.
    Returns:
        None

    """
    
    database_URL = 'sqlite:///{}'.format(database_filename)
    engine = create_engine(database_URL)
    df.to_sql('Categorized_Tweets', engine, index=False, if_exists="replace")
    engine.dispose()

def main():
    """Function to run the main steps in process_data.py. Loads, cleans & saves the data
    in preparation for multi-label classification of tweets.

    Args:
        None
    Returns:
        None

    """
    
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()