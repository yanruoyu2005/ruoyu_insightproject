"""This module contains the main functionalities of the WebApp."""
import pickle
import itertools
import matplotlib.pyplot as plt
import numpy as np
from numpy import loadtxt
from numpy import savetxt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.models import load_model

import config as cg
import strategy_interactive_plot


class RetrieveExisting:
    """Retrieve feature and label from existing dataset."""
    def __init__(self, number):
        self.number = number
        self.profile = cg.profile
        self.existing_x_test = cg.existing_x_test
        self.existing_y_test = cg.existing_y_test

    def get_feature(self):
        """Obtain test set feature, label and
        identify company index."""
        x_test_pd = pd.read_csv(self.existing_x_test)
        company_index = x_test_pd.iloc[:, 0][self.number]
        x_test_pd_feature = x_test_pd.iloc[:, 1:].iloc[:, :-1]
        x_test = np.asarray(x_test_pd_feature)
        y_test_pd = pd.read_csv(self.existing_y_test)
        y_test = np.asarray(y_test_pd.iloc[:, 1:])
        return x_test, y_test, company_index


class RetrieveCustomized:
    """Retrieve feature and label from customized dataset."""
    def __init__(self):
        self.profile = cg.profile
        self.key_words = cg.key_words
        self.key_words_pd = pd.read_csv(self.key_words)
        self.keywords_list = self.key_words_pd.iloc[0:,].values.reshape(1, -1).tolist()[0]
        self.feature_one_hot = cg.feature_one_hot
        self.customized_input = cg.customized_input
        self.normalization = cg.normalization
        self.profile_df = pd.read_json(self.profile)
        self.text = list(self.profile_df.iloc[:, 0])
        self.profile_df.columns = ['domain']
        self.feature_pd = pd.read_csv(self.feature_one_hot)
        self.feature_text_pd = self.feature_pd
        self.feature_text_pd['domain'] = self.profile_df['domain']
        self.feature_text_pd = self.feature_text_pd.iloc[:, 3:] # historical feature with profile

    def get_feature(self):
        """Return customized feature based on user input."""
        df = pd.read_csv(self.customized_input) # customized input
        column_names = self.feature_text_pd.columns
        df1 = pd.DataFrame(np.zeros([1, 58]), columns=column_names)
        # Assign input values to framework
        df1.investment_rounds = df.investment_rounds[0]
        df1.funding_rounds = df.funding_rounds[0]
        df1.funding_total_usd = df.funding_total_usd[0]
        df1.milestones = df.milestones[0]
        df1.relationships = df.relationships[0]
        df1.timeline = df.last_funding_year[0] - df.year_founded[0]
        df1.a = df[df.funding_types == 'Series a'].shape[0]
        df1.b = df[df.funding_types == 'Series b'].shape[0]
        df1.c = df[df.funding_types == 'Series c'].shape[0]
        df1.d = df[df.funding_types == 'Series d'].shape[0]
        df1.e = df[df.funding_types == 'Series e'].shape[0]
        df1.f = df[df.funding_types == 'Series f'].shape[0]
        df1.g = df[df.funding_types == 'Series g'].shape[0]
        df1.angel = df[df.funding_types == 'Angel'].shape[0]
        df1.convertible = df[df.funding_types == 'Convertible'].shape[0]
        df1.crowd = df[df.funding_types == 'Crowd'].shape[0]
        df1.crowd_equity = df[df.funding_types == 'Crowd Equity'].shape[0]
        df1.debt_round = df[df.funding_types == 'debt_round'].shape[0]
        df1.grant = df[df.funding_types == 'Grant'].shape[0]
        df1.partial = df[df.funding_types == 'Partial'].shape[0]
        df1.post_ipo_debt = df[df.funding_types == 'Post IPO Debt'].shape[0]
        df1.post_ipo_equity = df[df.funding_types == 'Post IPO Equity'].shape[0]
        df1.private_equity = df[df.funding_types == 'Private Equity'].shape[0]
        df1.secondary_market = df[df.funding_types == 'Secondary Market'].shape[0]
        df1.seed = df[df.funding_types == 'Seed'].shape[0]
        df1.unattributed = df[df.funding_types == 'Unattributed'].shape[0]
        df1.domain = df.domain[0]
        # df1 now contains one row of input feature
        new_feature = self.feature_text_pd.append(df1)
        return new_feature

    def feature_nlp(self, input_nlp):
        """Process feature by natural language processing."""
        column_trans = ColumnTransformer([('domain_bow',
                                           CountVectorizer(self.text,
                                                           analyzer='word',
                                                           binary=True,
                                                           decode_error='strict',
                                                           lowercase=True,
                                                           max_features=None,
                                                           max_df=0.80,
                                                           ngram_range=(1, 2),
                                                           preprocessor=None,
                                                           stop_words=['english',
                                                                       'company',
                                                                       'based',
                                                                       'founded',
                                                                       'and',
                                                                       'the',
                                                                       'in',
                                                                       'of',
                                                                       'for',
                                                                       'to',
                                                                       'is',
                                                                       'as',
                                                                       'inc',
                                                                       'with',
                                                                       'that',
                                                                       'was',
                                                                       'its',
                                                                       'it',
                                                                       'has',
                                                                       'on',
                                                                       'on',
                                                                       'by',
                                                                       'an',
                                                                       'which',
                                                                       'are'],
                                                           strip_accents=None,
                                                           tokenizer=None,
                                                           vocabulary=self.keywords_list),
                                           'domain')],
                                         remainder='passthrough')
        column_trans.fit(self.feature_text_pd)
        new_array1 = column_trans.transform(input_nlp).toarray()
        scaler_function = pickle.load(open(self.normalization, 'rb'))
        # input_np is general with domain-specific feature before normalization
        input_np = new_array1[570].reshape(1, -1)
        # x_test_input is normalized.
        x_test = scaler_function.transform(input_np)
        x_test_input = x_test
        return x_test_input, input_np


class SuccessForecast:
    """Forecast success and failure of startups."""
    def __init__(self, x_test, y_test, number):
        self.x_test = x_test
        self.y_test = y_test
        self.number = number
        self.deep_learning_model = cg.deep_learning_model

    def deep_learning_existing(self):
        """Perform deep learning on existing dataset."""
        classifier = load_model(self.deep_learning_model)
        y_pred = classifier.predict(self.x_test)
        y_pred1 = (y_pred > 0.5)
        if y_pred[self.number] > 0.5:
            prediction = 'Success'
        if y_pred[self.number] < 0.5:
            prediction = 'Failure'
        if self.y_test[self.number] == 1:
            actual = 'Success'
        if self.y_test[self.number] == 0:
            actual = 'Failure'
        return prediction, actual, y_pred1

    def deep_learning_customized(self):
        """Perform deep learning on customized dataset."""
        classifier = load_model(self.deep_learning_model)
        y_pred = classifier.predict(self.x_test)
        if y_pred[0][0] > 0.5:
            prediction = 'Success'
        else:
            prediction = 'Failure'
        print('Predicted outcome, ', prediction)
        return prediction


class ConfusionMatrix:
    """Generate a confusion matrix."""
    def __init__(self, y_test, y_pred1):
        self.y_test = y_test
        self.y_pred1 = y_pred1
        self.cm = confusion_matrix(self.y_test, self.y_pred1)
        self.classes = ['Failure', 'Success']

    def plot_confusion_matrix(self, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
        """Create a confusion matrix plot."""
        plt.imshow(self.cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        tick_marks = np.arange(len(self.classes))
        plt.xticks(tick_marks, self.classes, rotation=0)
        plt.yticks(tick_marks, self.classes)

        if normalize:
            self.cm = self.cm.astype('float') / self.cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')
        print(self.cm)

        thresh = self.cm.max() / 2.
        for i, j in itertools.product(range(self.cm.shape[0]), range(self.cm.shape[1])):
            plt.text(j, i, self.cm[i, j],
                     horizontalalignment="center",
                     color="white" if self.cm[i, j] > thresh else "black")
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        path = cg.path
        pic_name = path + '/static/assets/img/portfolio/confusion_matrix1.png'
        plt.savefig(pic_name)
        plt.close()


class Similarity:
    """Identify similar competitors."""
    def __init__(self, my_company_index=False, customized_feature=False):
        # self.target_company = target_company
        self.whole_feature = loadtxt(cg.whole_feature, delimiter=',')
        self.company_logo = pd.read_csv(cg.company_logo)
        if my_company_index == False:
            self.target_company = customized_feature
        else:
            self.target_company = self.whole_feature[my_company_index, :]
            self.target_company = np.reshape(self.target_company, (1, -1))

    def similarity(self):
        """Obtain the logos and logo urls of similar
        competitors."""
        similarity_list = []
        for single_feature in self.whole_feature:
            single_feature = np.reshape(single_feature, (1, -1))
            cosine_np = cosine_similarity(self.target_company, single_feature)
            similarity_list.append(cosine_np[0][0])
        cosine_df = pd.DataFrame(similarity_list)
        cosine_df.columns = ['cosine_score']
        cosine_df['numeric_index'] = list(range(self.whole_feature.shape[0]))
        cosine_df_sorted = cosine_df.sort_values(by=['cosine_score'], ascending=False)
        similarity_top = list(cosine_df_sorted['numeric_index'][0:6])
        print(similarity_top)

        df_url = self.company_logo.homepage_url
        url_top = []
        for numeric_index in similarity_top:
            my_url = df_url[numeric_index]
            if str(my_url) != 'nan':
                my_url = my_url[1:][:-1]
                url_top.append(my_url)
            else:
                url_top.append('nan')

        print(url_top)
        image_name_top = []
        for numeric_index in similarity_top:
            image_name = 'pic_'+ str(numeric_index+1) + '.png'
            image_name_top.append(image_name)

        return image_name_top, url_top


class Strategy:
    """Generate corporate strategies to reduce the risk of failure
    and improve the chance of success."""
    def __init__(self, my_company_index, my_prediction, my_actual, my_number):
        self.path = cg.path
        self.my_company_index = my_company_index
        self.my_prediction = my_prediction
        self.my_actual = my_actual
        self.my_number = my_number

    def generate_array(self):
        """Generate a strategy array represent all possible combinations of factors,
        x, y, z represents funding_total_usd, relationship and timeline."""
        factor_list = range(5, 41, 2)
        factor = np.asarray(factor_list)/10
        array = []
        for funding in factor:
            for relationship in factor:
                for timeline in factor:
                    array.append([funding, relationship, timeline])
        array = np.asarray(array)
        array_df = pd.DataFrame(array)
        deviation = []
        for i in range(array.shape[0]):
            deviation.append(abs(array[i, 0]-1) + abs(array[i, 1]-1) + abs(array[i, 2]-1))
        array_df['deviation'] = deviation
        array_df.to_csv(self.path + '/exploratory_data/factor_deviation.csv')
        return array

    def obtain_target(self, array):
        """Based on user input company ID, obtain its original feature."""
        df = pd.read_csv(cg.feature_one_hot)
        x_target = list(df.iloc[self.my_company_index, 3:])
        size = array.shape[0]
        target_list = []
        for i in range(size):
            target_list.append(x_target)
        target_array = np.asarray(target_list)
        return target_array, x_target

    def scale_array(self):
        """Generate a new array with hypothetical
        scaled feature based on the gradient of factors."""
        array = self.generate_array()
        target_array, x_target = self.obtain_target(array)
        x_target_temp = pd.DataFrame(x_target)
        x_target_temp.to_csv(self.path + '/exploratory_data/x_target_temp.csv')
        array[:, 0] = array[:, 0]*(x_target[2]+10000)
        array[:, 1] = array[:, 1]*(x_target[4]+1)
        array[:, 2] = array[:, 2]*(x_target[25]+0.1)
        new_array = array
        target_array[:, [2, 4, 25]] = new_array
        x_target_temp.to_csv(self.path + '/exploratory_data/x_target_temp.csv')
        file_name = self.path + '/exploratory_data/strategy_feature_temp.csv'
        savetxt(file_name, target_array, delimiter=',')
        return target_array

    def test_strategy(self, target_array):
        """Test whether proposed strategies succeed."""
        classifier = load_model(cg.deep_learning_model)
        data = loadtxt(cg.whole_feature, delimiter=',')
        x_test_number_nlp = list(data[self.my_company_index, :][57:])
        nlp_array = []
        for i in range(target_array.shape[0]):
            nlp_array.append(x_test_number_nlp)
        nlp_array = np.asarray(nlp_array)
        x_test0 = np.concatenate((target_array, nlp_array), axis=1)
        scaler_function = pickle.load(open(cg.scaler_file, 'rb'))
        x_test = scaler_function.fit_transform(x_test0)
        y_test = np.asarray([1]*target_array.shape[0])
        y_pred = classifier.predict(x_test)
        y_pred1 = (y_pred > 0.5)
        cm_strategy = confusion_matrix(y_test, y_pred1)
        print('confusion_matrix_new_strategy', cm_strategy)
        return cm_strategy, y_pred1

    def strategy(self):
        """Categorize the outcome of strategy and prioratize them."""
        strategy_outcome = 0
        if self.my_actual == 'Success':
            strategy_outcome = 2
            print('congrats')
        if self.my_actual == 'Failure':
            target_array = self.scale_array()
            cm_strategy, y_pred1 = self.test_strategy(target_array)
            if cm_strategy[1, 1] == 0:
                print('The generic strategy does not suit you, \
                    please contact us for developing customized strategy')
            else:
                strategy_outcome = 1
                print('We can help you succeed!')
                factor_deviation_df = pd.read_csv(self.path + \
                    '/exploratory_data/factor_deviation.csv')
                factor_deviation_df['strategy_prediction'] = y_pred1
                factor_deviation_df.to_csv(self.path + \
                    '/exploratory_data/factor_deviation_pre.csv')
                factor_true = factor_deviation_df[factor_deviation_df['strategy_prediction'] == True]
                factor_true_sort = factor_true.sort_values(by=['deviation'])
                factor_true_sort.to_csv(self.path + '/exploratory_data/factor_true_sort.csv')
                # create interactive plot
                strategy_interactive_plot.get_plot()
                data = factor_true_sort.iloc[0:10, 1:4]
                data.columns = ['Total Funding', 'Relationship', 'Timeline']
                strategy_list = []
                for i in range(10):
                    strategy = 'Strategy ' + str(i+1)
                    strategy_list.append(strategy)
                data.index = strategy_list
                plt.clf()
                plt.figure(figsize=(8, 10))
                title_name = 'Scaling Factors of Company ' + str(self.my_number)
                plt.rcParams['font.family'] = "serif"
                sns.set_context(font_scale=1.6)
                sns_plot = sns.heatmap(data, cmap='coolwarm',
                                       annot=True, fmt=".1f", annot_kws={'size':16})
                sns_plot.tick_params(labelsize=14)
                sns_plot.axes.set_title(title_name, fontsize=20)
                plt.yticks(rotation=0)
                plt.tight_layout()
                pic_name = self.path + '/static/assets/img/portfolio/strategy_sorted.png'
                plt.savefig(pic_name)
                plt.close()
        return strategy_outcome

    def strategy_customize(self, my_prediction1, my_input):
        """Propose strategies based on customized input."""
        strategy_outcome = 0
        if my_prediction1 == 'Success':
            strategy_outcome = 2
        else:
            array = self.generate_array()
            size = array.shape[0]
            target_list = []
            input1 = my_input.tolist()[0]
            print(input1)
            for i in range(size):
                target_list.append(input1)
            target_array = np.asarray(target_list)
            print(target_array.shape)

            array[:, 0] = array[:, 0]*(input1[2]+1)
            array[:, 1] = array[:, 1]*(input1[4]+0.1)
            array[:, 2] = array[:, 2]*(input1[25]+0.1)

            new_array = array
            target_array[:, [2, 4, 25]] = new_array

            file_name = self.path + '/exploratory_data/strategy_feature_temp.csv'
            savetxt(file_name, target_array, delimiter=',')

            scaler_function = pickle.load(open(cg.scaler_file, 'rb'))
            x_test = scaler_function.transform(target_array)

            classifier = load_model(cg.deep_learning_model)
            y_pred = classifier.predict(x_test)
            y_pred1 = (y_pred > 0.5)
            y_test = np.asarray([1]*target_array.shape[0])

            cm_strategy = confusion_matrix(y_test, y_pred1)
            print('confusion_matrix_new_strategy', cm_strategy)

            if cm_strategy[1, 1] == 0:
                print('The generic strategy does not suit you,\
                    please contact us for developing customized strategy')
            else:
                strategy_outcome = 1
                print('We can help you succeed!')

                factor_deviation_df = pd.read_csv(self.path + \
                    '/exploratory_data/factor_deviation.csv')
                factor_deviation_df['strategy_prediction'] = y_pred1
                factor_deviation_df.to_csv(self.path + '/exploratory_data/factor_deviation_pre.csv')
                factor_true = factor_deviation_df[factor_deviation_df['strategy_prediction'] == True]
                factor_true_sort = factor_true.sort_values(by=['deviation'])
                factor_true_sort.to_csv(self.path + '/exploratory_data/factor_true_sort.csv')
                # create interactive plot
                strategy_interactive_plot.get_plot()
                data = factor_true_sort.iloc[0:10, 1:4]
                data.columns = ['Total Funding', 'Relationship', 'Timeline']
                strategy_list = []
                for i in range(10):
                    strategy = 'Strategy ' + str(i+1)
                    strategy_list.append(strategy)
                data.index = strategy_list
                plt.clf()
                plt.figure(figsize=(8, 10))
                title_name = 'Scaling Factors of Target Company'
                plt.rcParams['font.family'] = "serif"
                sns.set_context(font_scale=1.6)
                sns_plot = sns.heatmap(data, cmap='coolwarm',
                                       annot=True, fmt=".1f", annot_kws={'size':16})
                sns_plot.tick_params(labelsize=14)
                sns_plot.axes.set_title(title_name, fontsize=20)
                plt.yticks(rotation=0)
                plt.tight_layout()
                pic_name = self.path + '/static/assets/img/portfolio/strategy_sorted.png'
                plt.savefig(pic_name)
                plt.close()
        return strategy_outcome
