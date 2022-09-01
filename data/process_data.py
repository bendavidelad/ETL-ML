import sys

import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(left=messages, right=categories, on='id', how='inner')
    return df


def clean_data(df):
    categories_df = build_categories_df(df)
    df = replace_categories_column_with_df(categories_df, df)
    df = df.drop_duplicates()
    return df


def replace_categories_column_with_df(categories_df, df):
    df = df.drop('categories', axis=1)
    df = pd.concat([df, categories_df], axis=1)
    return df


def build_categories_df(df):
    categories_split = df['categories'].str.split(";", expand=True)
    first_row = categories_split.head(1)
    categories_col_names = [cat[:-2] for cat in first_row.values[0]]
    categories_split.columns = categories_col_names
    for column in categories_split:
        categories_split[column] = categories_split[column].str[-1:]
        categories_split[column] = categories_split[column].astype(int)
    return categories_split


def save_data(df, database_filename):
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