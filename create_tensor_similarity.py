import pandas as pd
import torch
import numpy as np
import Levenshtein as lev
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class Tensor_similarity():
    def __init__(self, df_android, df_apple):
        self.google_play_df = df_android
        self.app_store_df = df_apple
        self.len_google_play = 1000
        self.len_app_store = 1000
        self.num_of_features = 5
        self.tensor_matching = torch.zeros((self.len_google_play, self.len_app_store, self.num_of_features))

        # my_tensor = torch.load('tensor.pt')
        # print(my_tensor.shape)
        # print(my_tensor[-10:, -10:])

    def pipeline(self):
        self.preparing_tensor_matching()
        self.create_csv()

    def preparing_tensor_matching(self):
        # TODO: create a torch for each batch according to batch size
        for i, google_play_app in self.google_play_df.head(self.len_google_play).iterrows():
            print(f'Google play batch: {i}')
            # TODO: Save torch according the batch size and index, i.e if i % 1000 == 0 then save torch and create a new one
            for j, app_store_app in self.app_store_df.head(self.len_app_store).iterrows():
                self.tensor_matching[i][j] = self.prep_two_apps_matching(google_play_app, app_store_app)

        torch.save(self.tensor_matching, 'tensor.pt')


    def prep_two_apps_matching(self, google_play_app, app_store_app):
        def get_string_differences(google_play_string, app_store_string):
            if not pd.isnull(app_store_string) and not pd.isnull(google_play_string):
                jaro = lev.ratio(app_store_string, google_play_string)
            else:
                jaro = 0
            return jaro

        def process_tfidf_similarity(google_description, apple_description):
            vectorizer = TfidfVectorizer()
            embeddings = vectorizer.fit_transform([apple_description, google_description])
            cosine_similarities = cosine_similarity(embeddings[0:1], embeddings[1:]).flatten()
            return cosine_similarities

        def releasedate_dist_similarity(google_releasedate, apple_releasedate):
            if np.isnat(google_releasedate) or np.isnat(apple_releasedate):
                return 241.040# the mean of differences that we calculated in the eda
            else:
                return np.absolute(google_releasedate - apple_releasedate).astype(int)



        tow_apps_matching = []
        # insert title similarity:
        tow_apps_matching.append(get_string_differences(google_play_app['title'], app_store_app['title']))

        # insert author similarity:
        tow_apps_matching.append(get_string_differences(google_play_app['author'], app_store_app['author']))

        # insert devsite similarity:
        tow_apps_matching.append(get_string_differences(google_play_app['devsite'], app_store_app['devsite']))

        # insert maincategory similarity:
        tow_apps_matching.append(get_string_differences(google_play_app['maincategory'], app_store_app['maincategory']))

        # # insert description similarity:
        # tow_apps_matching.append(process_tfidf_similarity(google_play_app['description'], app_store_app['description'])[0])

        # insert releasedate similarity:
        #tow_apps_matching.append(releasedate_dist_similarity(np.datetime64(google_play_app['releasedate']), np.datetime64(app_store_app['releasedate'])))

        # insert label 0/1:
        # TODO: make sure the labels stays as an int
        tow_apps_matching.append(1 if google_play_app['id'] == app_store_app['id_matched'] else 0)

        return torch.Tensor(tow_apps_matching)

    def create_csv(self):
        df = pd.DataFrame(self.tensor_matching)
        df.to_csv('tensor_matching.csv')




df_android = pd.read_csv('google_play_after_eda.csv', lineterminator='\n')
df_apple = pd.read_csv('app_store_after_eda.csv', lineterminator='\n')
T = Tensor_similarity(df_android, df_apple)
T.pipeline()

















