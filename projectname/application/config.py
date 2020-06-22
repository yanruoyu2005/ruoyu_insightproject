import os
import sys


path = os.path.dirname(os.path.realpath(sys.argv[0]))
profile =  path + '/exploratory_data/bt_c2to4_edited1.txt'
existing_x_test = path + '/exploratory_data/X_test_pd_v3_id.csv'
existing_y_test = path + '/exploratory_data//y_test_pd_v3.csv'
deep_learning_model = path + '/exploratory_data/pretrain_model_add_domain_binary_keywords_id.hdf5'
key_words = path + '/exploratory_data/key_words.csv'
feature_one_hot = path + '/exploratory_data/feature_one_hot1_0_1.csv'
customized_input = path + '/exploratory_data/input_df.csv'
normalization = path + '/exploratory_data/scaler.pkl'
whole_feature = path + '/exploratory_data/whole_feature.csv'
company_logo = path + '/exploratory_data/company_logo.csv'
scaler_file = path + '/exploratory_data/scaler.pkl'