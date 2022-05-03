import random
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import ndcg_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')
random.seed(128)


class AirbnbNewUserBookingsPipeline:
    """
    Data preparation and model prediction for 'Airbnb new users booking' competition on kaggle.
    Available methods:
    fit() - method calculates main statistics relying on train_users_2, sessions, countries, age_gender_bkts datasets
    transform() - method transforms train/test datasets correcting types, values and adding new features (total ~1000)
    train() - method trains the LGBMClassifier model relying on transforms dataset
    predict() - predict classes ranking @k
    """

    def __init__(self, importance_scores: pd.DataFrame = None, k: int = 5):
        """
        Pipeline parameters
        :param importance_scores: dataframe with importance data rank
        :param k:                 paramater for NDCG @k calculation and prediction data forming
        """
        # feature parameters
        self.lgb_model_weighted = None
        self.secs_elapsed_stats = pd.DataFrame()
        self.age_first_browser = pd.DataFrame()
        self.age_language = pd.DataFrame()
        self.age_affiliate_provider = pd.DataFrame()
        self.total_actions_counter = pd.DataFrame()
        self.action_detail_counter = pd.DataFrame()
        self.action_type_counter = pd.DataFrame()
        self.device_type_counter = pd.DataFrame()
        self.action_counter = pd.DataFrame()
        self.language_counter = pd.DataFrame()
        self.first_browser_counter = pd.DataFrame()
        self.first_device_type_counter = pd.DataFrame()
        self.signup_app_counter = pd.DataFrame()
        self.first_affiliate_tracked_counter = pd.DataFrame()
        self.affiliate_channel_counter = pd.DataFrame()
        self.language_age_group_counter = pd.DataFrame()
        self.first_brows_device_type_signup_counter = pd.DataFrame()
        self.first_affiliate_tracked_affiliate_channel_counter = pd.DataFrame()
        self.temporary_data = pd.DataFrame()
        # model parameters
        self.le = LabelEncoder()
        self.importance_scores = importance_scores
        self.k = k
        self.feature_selection = None
        self.lgb_model = None
        self.classes = None

    @classmethod
    def counter(cls,
                df: pd.DataFrame,
                first_group_col: str,
                second_group_col: str,
                ) -> pd.DataFrame:
        """
        Method for categorical features counter and comparing them with target one
        :param df:               input data frame
        :param first_group_col:  first categorical column
        :param second_group_col: second categorical column
        :return:                 grouped matrix of 'first * second' columns shape with counts
        """
        grouped_df = df.groupby([first_group_col, second_group_col])[second_group_col].agg('count').rename('count')
        return grouped_df.reset_index().pivot_table(values='count', index=second_group_col,
                                                    columns=first_group_col).fillna(0)

    @classmethod
    def df_correction(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Class method for general df correction
        :param df: input data frame
        :return:   corrected (unified) data frame
        """
        # making lower genders (unification)
        df['gender'] = df['gender'].apply(lambda x: x.lower())

        # there is no canadian language... besides, we are going to use levenshtein distance from countries dataset
        df.loc[df['language'] == 'ca', 'language'] = 'en'

        # correct format and type of time
        df['timestamp_first_active'] = pd.to_datetime(df['timestamp_first_active'], format='%Y%m%d%H%M%S')
        df['date_account_created'] = df['date_account_created'].astype('datetime64[ns]')
        df['date_first_booking'] = df['date_first_booking'].astype('datetime64[ns]')

        # correct age values
        df.loc[df['age'] > 1900, 'age'] = 2015 - df['age']
        df.loc[(df['age'] > 100) | (df['age'] < 15), 'age'] = np.nan

        # create age buckets feature
        df['age_group'] = np.nan
        df.loc[df['age'] < 4, 'age_group'] = '0-4'
        df.loc[(df['age'] >= 5) & (df['age'] < 9), 'age_group'] = '5-9'
        df.loc[(df['age'] >= 10) & (df['age'] < 14), 'age_group'] = '10-14'
        df.loc[(df['age'] >= 15) & (df['age'] < 19), 'age_group'] = '15-19'
        df.loc[(df['age'] >= 20) & (df['age'] < 24), 'age_group'] = '20-24'
        df.loc[(df['age'] >= 25) & (df['age'] < 29), 'age_group'] = '25-29'
        df.loc[(df['age'] >= 30) & (df['age'] < 34), 'age_group'] = '30-34'
        df.loc[(df['age'] >= 35) & (df['age'] < 39), 'age_group'] = '35-39'
        df.loc[(df['age'] >= 40) & (df['age'] < 44), 'age_group'] = '40-44'
        df.loc[(df['age'] >= 45) & (df['age'] < 49), 'age_group'] = '45-49'
        df.loc[(df['age'] >= 50) & (df['age'] < 54), 'age_group'] = '50-54'
        df.loc[(df['age'] >= 55) & (df['age'] < 59), 'age_group'] = '55-59'
        df.loc[(df['age'] >= 60) & (df['age'] < 64), 'age_group'] = '60-64'
        df.loc[(df['age'] >= 65) & (df['age'] < 69), 'age_group'] = '65-69'
        df.loc[(df['age'] >= 70) & (df['age'] < 74), 'age_group'] = '70-74'
        df.loc[(df['age'] >= 75) & (df['age'] < 79), 'age_group'] = '75-79'
        df.loc[(df['age'] >= 80) & (df['age'] < 84), 'age_group'] = '80-84'
        df.loc[(df['age'] >= 85) & (df['age'] < 89), 'age_group'] = '85-89'
        df.loc[(df['age'] >= 90) & (df['age'] < 94), 'age_group'] = '90-94'
        df.loc[(df['age'] >= 95) & (df['age'] < 99), 'age_group'] = '95-99'
        df.loc[df['age'] >= 100, 'age_group'] = '100+'

        return df

    @staticmethod
    def score_model(train_model: callable,
                    X_score: pd.DataFrame,
                    y_score: np.ndarray,
                    k: int = 5) -> float:
        """
        NDCG score function @k
        :param train_model: Input model supported .predict_proba()
        :param X_score:     X data for prediction and followed by scoring
        :param y_score:     labels for scoring
        :param k:           NDCG-score rank @k
        :return:            NDCG-score value
        """
        return ndcg_score(pd.get_dummies(y_score).to_numpy(), train_model.predict_proba(X_score), k=k)

    def fit(self,
            input_df: pd.DataFrame,
            sessions_df: pd.DataFrame,
            countries_df: pd.DataFrame,
            age_gender_bkts_df: pd.DataFrame):
        """
        Calculate and save properties from train dataset
        :param input_df:           input train dataset
        :param sessions_df:        sessions dataset
        :param countries_df:       countries dataset
        :param age_gender_bkts_df: countries info dataset
        """

        # Make main types and values correction for correct statistics calculation
        df = input_df.copy()
        df = self.df_correction(df)

        # Fitting of weighted boosting model
        X_train_data = df[['gender', 'signup_flow', 'first_browser',
                           'first_affiliate_tracked', 'first_device_type',
                           'signup_method', 'affiliate_channel',
                           'affiliate_provider', 'language', 'signup_app',
                           ]]
        y_train_data = self.le.fit_transform(df['country_destination'])
        X_train, X_valid, y_train, y_valid = train_test_split(X_train_data, y_train_data,
                                                              test_size=0.2, random_state=128)
        string_cols = X_train.select_dtypes(exclude=[np.number]).columns
        X_train[string_cols] = X_train[string_cols].astype('category')
        X_valid[string_cols] = X_valid[string_cols].astype('category')
        self.classes = self.le.inverse_transform(np.unique(y_train_data))

        lgb_params_weighted = {"n_estimators": 1500,
                               "max_depth": 5,
                               "n_jobs": -1,
                               "reg_lambda": 5,
                               "class_weight": 'balanced',
                               "random_state": 128,
                               }
        self.lgb_model_weighted = lgb.LGBMClassifier(**lgb_params_weighted)
        self.lgb_model_weighted.fit(X_train, y_train,
                                    early_stopping_rounds=100,
                                    eval_set=[(X_train, y_train), (X_valid, y_valid)],
                                    verbose=0)

        # Statistical properties fitting (general statistics)
        self.secs_elapsed_stats = sessions_df.groupby('user_id').agg(min=('secs_elapsed', np.min),
                                                                     max=('secs_elapsed', np.max),
                                                                     mean=('secs_elapsed', np.mean),
                                                                     median=('secs_elapsed', np.median),
                                                                     std=('secs_elapsed', np.std),
                                                                     ).fillna(0).add_prefix('secs_elapsed_')

        self.age_first_browser = df.groupby('first_browser')[['age']].agg(np.median).reset_index()
        self.age_language = df.groupby('language')[['age']].agg(np.median).reset_index()
        self.age_affiliate_provider = df.groupby('affiliate_provider')[['age']].agg(np.median).reset_index()

        # count statistics from sessions_df
        self.total_actions_counter = sessions_df.groupby('user_id').agg('count').add_suffix('_count')
        self.action_detail_counter = self.counter(df=sessions_df,
                                                  first_group_col='action_detail',
                                                  second_group_col='user_id').add_prefix('action_detail_')
        self.action_type_counter = self.counter(df=sessions_df,
                                                first_group_col='action_type',
                                                second_group_col='user_id').add_prefix('action_type_')
        self.device_type_counter = self.counter(df=sessions_df,
                                                first_group_col='device_type',
                                                second_group_col='user_id').add_prefix('device_type_')
        self.action_counter = self.counter(df=sessions_df,
                                           first_group_col='action',
                                           second_group_col='user_id').add_prefix('action_')

        # count statistics relying on target
        self.language_counter = self.counter(df=df,
                                             first_group_col='country_destination',
                                             second_group_col='language').add_prefix('language_')
        self.first_browser_counter = self.counter(df=df,
                                                  first_group_col='country_destination',
                                                  second_group_col='first_browser').add_prefix('first_browser_')
        self.first_device_type_counter = self.counter(df=df,
                                                      first_group_col='country_destination',
                                                      second_group_col='first_device_type').add_prefix(
            'first_device_type_')
        self.signup_app_counter = self.counter(df=df,
                                               first_group_col='country_destination',
                                               second_group_col='signup_app').add_prefix('signup_app_')
        self.first_affiliate_tracked_counter = self.counter(df=df,
                                                            first_group_col='country_destination',
                                                            second_group_col='first_affiliate_tracked').add_prefix(
            'first_affiliate_tracked_')
        self.affiliate_channel_counter = self.counter(df=df,
                                                      first_group_col='country_destination',
                                                      second_group_col='affiliate_channel').add_prefix(
            'affiliate_channel_')

        # groups of features
        self.language_age_group_counter = df.pivot_table(index=['language', 'age_group'],
                                                         columns='country_destination',
                                                         values='id',
                                                         aggfunc='count').fillna(0).add_prefix('language_age_group_')
        self.first_brows_device_type_signup_counter = df.pivot_table(
            index=['first_browser', 'first_device_type', 'signup_app'],
            columns='country_destination',
            values='id',
            aggfunc='count').fillna(0).add_prefix('first_brows_device_type_signup_')
        self.first_affiliate_tracked_affiliate_channel_counter = df.pivot_table(
            index=['first_affiliate_tracked', 'affiliate_channel'],
            columns='country_destination',
            values='id',
            aggfunc='count').fillna(0).add_prefix('first_affiliate_tracked_affiliate_channel_')

        # fitting of tables with targets information
        targets = pd.merge(countries_df,
                           age_gender_bkts_df.groupby('country_destination')[['population_in_thousands']].agg(np.sum),
                           how='left', on='country_destination')
        self.temporary_data = pd.merge(df, targets.drop(columns='destination_language '), how='left',
                                       on='country_destination').dropna(subset=['population_in_thousands'])

    def transform(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """
        Method transforms dataset according to previous fits
        :param input_df: input dataset
        :return:         transformed dataset with added features
        """
        df = input_df.copy()
        # correct types and some values of data frame
        df = self.df_correction(df)

        # modelled feature
        X_weighted_feature = df[['gender', 'signup_flow', 'first_browser',
                                 'first_affiliate_tracked', 'first_device_type',
                                 'signup_method', 'affiliate_channel',
                                 'affiliate_provider', 'language', 'signup_app',
                                 ]]
        string_cols = X_weighted_feature.select_dtypes(exclude=[np.number]).columns
        X_weighted_feature[string_cols] = X_weighted_feature[string_cols].astype('category')
        df['weighted_feature'] = self.lgb_model_weighted.predict(X_weighted_feature)

        # time features
        df['year_account_created'] = df['date_account_created'].dt.year.astype('int32')
        df['month_account_created'] = df['date_account_created'].dt.month.astype('int32')
        df['day_account_created'] = df['date_account_created'].dt.day.astype('int32')
        df['weekday_account_created'] = df['date_account_created'].dt.weekday
        df['year_first_active'] = df['timestamp_first_active'].dt.year.astype('int32')
        df['month_first_active'] = df['timestamp_first_active'].dt.month.astype('int32')
        df['day_first_active'] = df['timestamp_first_active'].dt.day.astype('int32')
        df['weekday_first_active'] = df['timestamp_first_active'].dt.weekday

        # secs_elapsed time statistics
        df = pd.merge(df, self.secs_elapsed_stats, how='left', left_on='id', right_on='user_id')

        # device groups
        df['global_device'] = np.nan
        df.loc[np.isin(df['first_device_type'],
                       ['Mac Desktop', 'Windows Desktop', 'Desktop (Other)']), 'global_device'] = 'pc'
        df.loc[np.isin(df['first_device_type'], ['iPhone', 'Android Tablet', 'iPad', 'Android Phone',
                                                 'SmartPhone (Other)']), 'global_device'] = 'mobile'
        df.loc[np.isin(df['first_device_type'], ['Other/Unknown']), 'global_device'] = 'other'

        # grouped statistics from train dataset
        df = pd.merge(df, self.age_first_browser, how='left', on='first_browser', suffixes=('', '_first_browser'))
        df = pd.merge(df, self.age_language, how='left', on='language', suffixes=('', '_language'))
        df = pd.merge(df, self.age_affiliate_provider, how='left', on='affiliate_provider',
                      suffixes=('', '_affiliated_provider'))

        df = pd.merge(df, self.total_actions_counter, how='left', left_on='id', right_on='user_id')

        df = pd.merge(df, self.action_type_counter, how='left', left_on='id', right_on='user_id')
        df = pd.merge(df, self.action_detail_counter, how='left', left_on='id', right_on='user_id')
        df = pd.merge(df, self.action_counter, how='left', left_on='id', right_on='user_id')
        df = pd.merge(df, self.device_type_counter, how='left', left_on='id', right_on='user_id')

        df = pd.merge(df, self.language_counter, how='left', on='language')
        df = pd.merge(df, self.first_browser_counter, how='left', on='first_browser')
        df = pd.merge(df, self.first_device_type_counter, how='left', on='first_device_type')
        df = pd.merge(df, self.signup_app_counter, how='left', on='signup_app')
        df = pd.merge(df, self.first_affiliate_tracked_counter, how='left', on='first_affiliate_tracked')
        df = pd.merge(df, self.affiliate_channel_counter, how='left', on='affiliate_channel')

        df = pd.merge(df, self.language_age_group_counter, how='left', on=['language', 'age_group'])
        df = pd.merge(df, self.first_brows_device_type_signup_counter, how='left',
                      on=['first_browser', 'first_device_type', 'signup_app'])
        df = pd.merge(df, self.first_affiliate_tracked_affiliate_channel_counter, how='left',
                      on=['first_affiliate_tracked', 'affiliate_channel'])

        # features from targets tables
        index_list = ['language', 'first_browser', 'first_device_type', 'signup_app', 'first_affiliate_tracked',
                      'affiliate_channel']
        values_list = ['population_in_thousands', 'language_levenshtein_distance', 'destination_km2', 'distance_km',
                       'lng_destination', 'lat_destination']
        for index in index_list:
            for value in values_list:
                aggregated_table = self.temporary_data.pivot_table(index=index,
                                                                   columns='country_destination',
                                                                   values=value,
                                                                   aggfunc='median').fillna(0).add_prefix(
                    f'{index}_{value}_')
                df = pd.merge(df, aggregated_table, how='left', on=index)

        return df

    def train(self,
              df: pd.DataFrame,
              importance_threshold: float = 0.):
        """
        Method for training the model
        :param df:                   input dataframe
        :param importance_threshold: threshold value for feature selecting (mean value of 'permutated' data)
        """
        # tuned params
        lgb_params = {'boosting_type': 'gbdt',
                      'objective': 'multiclass',
                      'n_estimators': 1500,
                      'n_jobs': -1,
                      'random_state': 128,
                      'max_depth': 9,
                      'min_samples_leaf': 17,
                      'min_sum_hessian_per_leaf': 0.00625559766473137,
                      'feature_fraction': 0.9119246927135007,
                      'reg_lambda': 8.403115659399436,
                      'reg_alpha': 1.1431060092054413,
                      'min_data_per_group': 137,
                      'learning_rate': 0.016171466163550918,
                      'num_leaves': 29}

        X_train_data = df.drop(columns=['id', 'country_destination', 'date_first_booking', 'date_account_created',
                                        'timestamp_first_active'])
        y_train_data = self.le.fit_transform(df['country_destination'])
        self.classes = self.le.inverse_transform(np.unique(y_train_data))
        X_train, X_valid, y_train, y_valid = train_test_split(X_train_data, y_train_data, test_size=0.2,
                                                              random_state=128)
        string_cols = X_train.select_dtypes(exclude=[np.number]).columns
        X_train[string_cols] = X_train[string_cols].astype("category")
        X_valid[string_cols] = X_valid[string_cols].astype("category")

        if self.importance_scores is not None:
            self.feature_selection = self.importance_scores.loc[
                self.importance_scores['importance-mean'] > importance_threshold, 'features'].values
            X_train = X_train[self.feature_selection]
            X_valid = X_valid[self.feature_selection]
        else:
            print(f'WARNING! There is no importance_score file! The model will train on {X_train.shape[1]} features')

        self.lgb_model = lgb.LGBMClassifier(**lgb_params)
        self.lgb_model.fit(X_train, y_train,
                           early_stopping_rounds=100,
                           eval_set=[(X_train, y_train), (X_valid, y_valid)],
                           verbose=100)

        train_score = AirbnbNewUserBookingsPipeline.score_model(self.lgb_model, X_train, y_train, k=self.k)
        valid_score = AirbnbNewUserBookingsPipeline.score_model(self.lgb_model, X_valid, y_valid, k=self.k)
        print('=' * 70)
        print(f'Training succeeded with {X_train.shape[1]} features!')
        print(f'Train score: {train_score:.4}, Valid score: {valid_score:.4}')

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Method for forming final prediction @k
        :param df: dataset with data for prediction
        :return:   predicted table data
        """
        if self.importance_scores is not None:
            X_test = df[self.feature_selection]
        else:
            X_test = df.drop(columns=['id', 'date_first_booking', 'date_account_created', 'timestamp_first_active'])
            print(f'WARNING! There is no importance_score file! The model will train on {X_test.shape[1]} features')

        string_cols = X_test.select_dtypes(exclude=[np.number]).columns
        X_test[string_cols] = X_test[string_cols].astype("category")
        y_predict = self.lgb_model.predict_proba(X_test)

        result = pd.DataFrame(columns=['id', 'country'])
        prediction_df = pd.DataFrame(data=y_predict, columns=self.classes)

        for line in range(len(y_predict)):
            result = result.append(pd.DataFrame({'id': [df['id'].iloc[line] for _ in range(self.k)],
                                                 'country': prediction_df.iloc[line].nlargest(self.k).index,
                                                 }), ignore_index=True)
        print(f'Prediction succeeded with {X_test.shape[1]} features!')

        return result


if __name__ == '__main__':
    test = pd.read_csv('test_users.csv')
    train = pd.read_csv('train_users_2.csv')
    age_gender_bkts_train = pd.read_csv('age_gender_bkts.csv')
    countries_train = pd.read_csv('countries.csv')
    sessions_train = pd.read_csv('sessions.csv')

    importance_scores = pd.read_csv("importance_scores.csv", index_col=0)

    pipeline = AirbnbNewUserBookingsPipeline(importance_scores)
    pipeline.fit(train, sessions_train, countries_train, age_gender_bkts_train)
    train = pipeline.transform(train)
    test = pipeline.transform(test)
    pipeline.train(train)
    result = pipeline.predict(test)

    result.to_csv('prediction_result.csv', index=False, encoding='utf-8')
