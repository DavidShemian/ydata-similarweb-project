import pandas as pd
import numpy as np
import tldextract
import dateparser
from cleanco import prepare_terms, basename
import unidecode
import re
import string
from sklearn.model_selection import train_test_split

import logging
log = logging.getLogger("preproccessing-logger")

class Preprocessing:
    def __init__(self, df):
        self.df_apps_match = df
        self.google_play_df = self.df_apps_match[self.df_apps_match['store'].values.astype(
            int) == 0]
        self.app_store_df = self.df_apps_match[self.df_apps_match['store'].values.astype(
            int) == 1]
        # self.google_play_df_after_eda = pd.DataFrame()
        # self.app_store_df_after_eda = pd.DataFrame()
        # self.google_play_df_after_eda_10 = pd.DataFrame()
        # self.app_store_df_after_eda_10 = pd.DataFrame()
        self.df_after_preprocessing = pd.DataFrame()

    def pipeline(self):
        # self.clean_columns()
        # self.create_matched_tables() # todo: for the tensor

        self.add_id()
        self.preprocessing_maincategory()
        self.preprocessing_titles()
        self.preprocessing_author()
        self.preprocessing_devsite()
        # self.preprocessing_releasedate()
        self.preprocessing_description()
        self.add_matching()
        self.divide_data_80_20()
        self.save_csvs()

    def add_id(self):
        log.info('Adding id')

        self.df_after_preprocessing["id"] = self.df_apps_match["id"]

    def preprocessing_maincategory(self):
        log.info('preprocessing_maincategory')

        # Change from apple catagories ids to string catagories
        self.df_after_preprocessing["apple_maincategory"] = (
            self.df_apps_match[df_apps_match["store"] == 1]
            .loc[:, "maincategory"]
            .replace(
                [
                    "6000",
                    "6001",
                    "6002",
                    "6003",
                    "6004",
                    "6005",
                    "6006",
                    "6007",
                    "6008",
                    "6009",
                    "6010",
                    "6011",
                    "6012",
                    "6013",
                    "6014",
                    "6015",
                    "6016",
                    "6017",
                    "6018",
                    "6020",
                    "6021",
                    "6023",
                    "6024",
                    "6025",
                    "6026",
                    "6027",
                ],
                [
                    "Business",
                    "Weather",
                    "Utilities",
                    "Travel",
                    "Sports",
                    "Social Networking",
                    "Reference",
                    "Productivity",
                    "Photo Video",
                    "News",
                    "Navigation",
                    "Music",
                    "Lifestyle",
                    "Health and Fitness",
                    "Games",
                    "Finance",
                    "Entertainment",
                    "Education",
                    "Books",
                    "Medical",
                    "Magazines and Newspapers",
                    "Food and Drink",
                    "Shopping",
                    "Stickers",
                    "Developer Tools",
                    "Graphics and Design",
                ],
            )
        )

        # Change from google play catagories to apple catagories
        self.df_after_preprocessing["google_maincategory"] = (
            self.df_apps_match[df_apps_match["store"] == 0]
            .loc[:, "maincategory"]
            .replace(
                [
                    "BOOKS_AND_REFERENCE",
                    "BUSINESS",
                    "EDUCATION",
                    "ENTERTAINMENT",
                    "FINANCE",
                    "FOOD_AND_DRINK",
                    "GAME_ACTION",
                    "GAME_ADVENTURE",
                    "GAME_ARCADE",
                    "GAME_BOARD",
                    "GAME_CARD",
                    "GAME_CASINO",
                    "GAME_CASUAL",
                    "GAME_EDUCATIONAL",
                    "GAME_MUSIC",
                    "GAME_PUZZLE",
                    "GAME_RACING",
                    "GAME_ROLE_PLAYING",
                    "GAME_SIMULATION",
                    "GAME_SPORTS",
                    "GAME_STRATEGY",
                    "GAME_TRIVIA",
                    "GAME_WORD",
                    "HEALTH_AND_FITNESS",
                    "LIFESTYLE",
                    "MAPS_AND_NAVIGATION",
                    "MEDICAL",
                    "MUSIC_AND_AUDIO",
                    "NEWS_AND_MAGAZINES",
                    "PHOTOGRAPHY",
                    "PRODUCTIVITY",
                    "SHOPPING",
                    "SOCIAL",
                    "SPORTS",
                    "SPORTS_GAMES",
                    "TRAVEL_AND_LOCAL",
                    "VIDEO_PLAYERS",
                    "WEATHER",
                ],
                [
                    "Books",
                    "Business",
                    "Education",
                    "Entertainment",
                    "Finance",
                    "Food and Drink",
                    "Games",
                    "Games",
                    "Games",
                    "Games",
                    "Games",
                    "Games",
                    "Games",
                    "Games",
                    "Games",
                    "Games",
                    "Games",
                    "Games",
                    "Games",
                    "Games",
                    "Games",
                    "Games",
                    "Games",
                    "Health and Fitness",
                    "Lifestyle",
                    "Navigation",
                    "Medical",
                    "Music",
                    "Magazines and Newspapers",
                    "Photo Video",
                    "Productivity",
                    "Shopping",
                    "Social Networking",
                    "Sports",
                    "Sports",
                    "Travel",
                    "Photo Video",
                    "Weather",
                ],
            )
        )

    def preprocessing_titles(self):
        log.info('preprocessing_titles')

        # lower case the titles and seperate the title
        def create_title(titles):
            # todo: ask davis if need it also for athuor
            return [
                title.lower()
                .strip()
                .partition(":")[0]
                .partition("-")[0]
                .partition(" ")[0]
                for title in titles
            ]

        self.df_after_preprocessing["title"] = create_title(self.df_apps_match["title"])

    def preprocessing_author(self):
        log.info('preprocessing_author')

        def create_author(authors):
            terms = prepare_terms()
            # Running twice in order to remove multiple endings, i.e Co., Ltd.
            authors = [
                basename(
                    author.lower().strip(), terms, prefix=True, middle=True, suffix=True
                )
                for author in authors
            ]
            authors = [
                basename(
                    author, terms, prefix=True, middle=True, suffix=True
                ).partition(" ")[0]
                for author in authors
            ]
            return authors

        self.df_after_preprocessing["author"] = create_author(self.df_apps_match["author"])

        def create_devsite(devsites):
            return [
                tldextract.extract(devsite.lower().strip()).domain
                for devsite in devsites
            ]

        self.df_after_preprocessing["devsite"] = create_devsite(self.df_apps_match["devsite"])

    def preprocessing_releasedate(self):
        log.info('preprocessing_releasedate')

        def parse_date(date):
            if not isinstance(date, str):
                # always nan values
                return

            return dateparser.parse(date)

        self.google_play_df_after_eda["releasedate"] = pd.to_datetime(
            self.google_play_df["releasedate"].apply(parse_date), errors="coerce"
        )
        # self.google_play_df['releasedate'].apply(parse_date).values.astype('datetime64[D]')
        self.app_store_df_after_eda["releasedate"] = pd.to_datetime(
            self.app_store_df["releasedate"].apply(parse_date), errors="coerce"
        )

    def preprocessing_description(self):  # todo: make it better..
        log.info('preprocessing_description')

        def create_descriptions(descriptions):
            return [
                unidecode.unidecode(re.sub(r"\d+", "", description))
                .lower()
                .translate(str.maketrans("", "", string.punctuation))
                .strip()
                for description in descriptions
            ]

        self.df_after_preprocessing["description"] = create_descriptions(self.df_apps_match["description"])

    def add_matching(self):
        log.info('add_matching')

        self.df_after_preprocessing["id_matched"] = create_descriptions(self.df_apps_match["id_matched"])

    def divide_data_80_20(self):
        log.info('divide_data_80_20')

        for i, google_play_app in self.google_play_df.iterrows():
            if i % 10 == 0:
                self.google_play_df_after_eda_10 = (
                    self.google_play_df_after_eda_10.append(
                        google_play_app, ignore_index=True
                    )
                )
                self.app_store_df_after_eda_10 = self.app_store_df_after_eda_10.append(
                    self.app_store_df_after_eda.loc[
                        self.app_store_df_after_eda["id_matched"]
                        == google_play_app["id"]
                    ]
                )
                self.google_play_df_after_eda.drop(i, inplace=True)
                self.app_store_df_after_eda.drop(
                    self.app_store_df_after_eda[
                        self.app_store_df_after_eda["id_matched"]
                        == google_play_app["id"]
                    ].index,
                    inplace=True,
                )

    def train_test_split(self):
        log.info('train_test_split')
        
        # Shuffle dataset 
        shuffle_df = self.df_after_preprocessing.sample(frac=1)

        # get 10% of data
        test_size = int(0.1 * len(df))

        test_set_first_part = shuffle_df[:train_size]
        train_set_second_part = self.df_after_preprocessing.loc(self.df_after_preprocessing["id_matched"] == train_set_first_part["id"])

        self.train_data = pd.concat([train_set_first_part, train_set_second_part])
        self.test_data  = self.df_after_preprocessing[self.df_after_preprocessing["id"] != self.train_data["id"]]
        

    def save_csvs(self):
        self.google_play_df_after_eda.to_csv(
            r"google_play_after_eda.csv", index=False, header=True
        )
        self.app_store_df_after_eda.to_csv(
            r"app_store_after_eda.csv", index=False, header=True
        )
        self.google_play_df_after_eda_10.to_csv(
            r"google_play_after_eda_10.csv", index=False, header=True
        )
        self.app_store_df_after_eda_10.to_csv(
            r"app_store_after_eda_10.csv", index=False, header=True
        )

    def clean_columns(self):  # stay with columns that no Nan in app_store
        for column in self.app_store_df.columns:
            if self.app_store_df[column].isnull().all():
                self.app_store_df = self.app_store_df.drop(column, axis=1)
                self.google_play_df = self.google_play_df.drop(column, axis=1)

    def create_matched_tables(self):
        self.google_play_df = self.google_play_df.rename(
            columns={"id_matched": "ios_id"}
        )
        self.app_store_df = self.app_store_df.rename(
            columns={"id_matched": "android_id"}
        )
        self.matched_app_store = pd.merge(
            self.app_store_df,
            self.google_play_df,
            left_on="android_id",
            right_on="id",
            suffixes=("_app_store", "_google_play"),
        )
        self.matched_google_play = pd.merge(
            self.google_play_df,
            self.app_store_df,
            left_on="ios_id",
            right_on="id",
            suffixes=("_google_play", "_app_store"),
        )


df = pd.read_csv("../data/apps_matching.csv", low_memory=False)
print(df.shape)
# P = Preprocessing(df)
# P.pipeline()
