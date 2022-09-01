import sys

import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    loads messages and categories from csv and merges them
    :param messages_filepath:  the path of the messages.csv
    :type messages_filepath: str
    :param categories_filepath: the path of categories.csv
    :type categories_filepath: str
    :return: the merged dataframe
    :rtype: pd.Dataframe
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(left=messages, right=categories, on='id', how='inner')
    return df


def clean_data(df):
    """
    Cleans the given dataFrame, replacing categories column with categories df of 0 and 1
    and drops duplicates
    :param df: given df
    :type df: pd.DataFrame
    :return: the cleaned dataFrame
    :rtype: pd.DataFrame
    """
    categories_df = build_categories_df(df)
    df = replace_categories_column_with_df(categories_df, df)
    df = df.drop_duplicates()
    return df


def replace_categories_column_with_df(categories_df, df):
    """
    replaces the categories column with the categories dataFrame
    :param categories_df: the dataframe to replace
    :type categories_df: pd.DataFrame
    :param df: the dataframe that contains the column to replace
    :type df: pd.DataFrame
    :return: the dataframe after the replacement
    :rtype: pd.DataFrame
    """
    df = df.drop('categories', axis=1)
    df = pd.concat([df, categories_df], axis=1)
    return df


def build_categories_df(df):
    """
    build a dataframe that at the columns has all the categories and the values inside are either
    0 or 1 for each row
    :param df: dataframe that has a categories column
    :type df: pd.DataFrame
    :return: the categories dataframe
    :rtype: pd.DataFrame
    """
    categories_split = df['categories'].str.split(";", expand=True)
    first_row = categories_split.head(1)
    categories_col_names = [cat[:-2] for cat in first_row.values[0]]
    categories_split.columns = categories_col_names
    for column in categories_split:
        categories_split[column] = categories_split[column].str[-1:]
        categories_split[column] = categories_split[column].astype(int)
    return categories_split


def save_data(df, database_filename):
    """
    Saves the df into a database
    :param df: the df to save
    :type df: pd.DataFrame
    :param database_filename: the name of db file
    :type database_filename: str
    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('messages', engine, index=False)


def main():
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
        print('Please provide the filepaths of the messages and categories ' \
              'datasets as the first and second argument respectively, as ' \
              'well as the filepath of the database to save the cleaned data ' \
              'to as the third argument. \n\nExample: python process_data.py ' \
              'disaster_messages.csv disaster_categories.csv ' \
              'DisasterResponse.db')


if __name__ == '__main__':
    main()