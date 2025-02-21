import pmlb
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import time
from tpot import TPOTClassifier, TPOTRegressor
from prettytable import PrettyTable
import random
import pdb


generations=3
population=10
max_time_mins=1

def run_clf_experiments():
    result_table = PrettyTable()
    result_table.field_names = ['dataset', 'operation', 'n_rows', 'n_cols', 'test_size',
                                'random_state',
                                'default_train_accuracy', 'default_test_accuracy',
                                'tpot_train_accuracy', 'topt_test_accuracy',
                                'default_time', 'tpot_time']
    result_table.float_format = '.2'

    clf_datasets = pmlb.classification_dataset_names
    i = 1
    for ds in clf_datasets:
        print(f'Working on {ds}, [{i}/{len(clf_datasets)}]')
        row = []
        X, y = pmlb.fetch_data(ds, return_X_y=True, local_cache_dir='/home/vinayd/nwork/pmlb_cache/')
        row.append(ds)
        row.append('classification')
        row.append(X.shape[0])
        row.append(X.shape[1])

        test_size = 0.2
        row.append(test_size)

        random_state = int(random.random() * 4294967295)
        row.append(random_state)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        default_rf = RandomForestClassifier(n_estimators=100, random_state=random_state)
        tpot_rf = TPOTClassifier(template='RandomForestClassifier', generations=generations, population_size=population,
                                    verbosity=2, random_state=random_state, n_jobs=-1, max_time_mins=max_time_mins)
        start_time = time.time()
        default_rf.fit(X_train, y_train)
        end_time = time.time()
        default_time = end_time - start_time

        start_time = time.time()
        tpot_rf.fit(X_train, y_train)
        end_time = time.time()
        tpot_time = end_time - start_time

        #pdb.set_trace()
        tpot_rf_params = tpot_rf.fitted_pipeline_[0].get_params()

        default_train_accuracy = default_rf.score(X_train, y_train)
        default_test_accuracy = default_rf.score(X_test, y_test)

        tpot_train_accuracy = tpot_rf.score(X_train, y_train)
        tpot_test_accuracy = tpot_rf.score(X_test, y_test)
       
        row.append(default_train_accuracy)
        row.append(default_test_accuracy)
        row.append(tpot_train_accuracy)
        row.append(tpot_test_accuracy)

        row.append(default_time)
        row.append(tpot_time)
        result_table.add_row(row)
        #print(f'Training time = {end_time - start_time}')
        #break
        with open('classification_result.csv', 'w') as fp:
            fp.write(result_table.get_csv_string())
        i += 1

def run_reg_experiments():
    result_table = PrettyTable()
    result_table.field_names = ['dataset', 'operation', 'n_rows', 'n_cols', 'test_size',
                                'random_state',
                                'default_train_accuracy', 'default_test_accuracy',
                                'tpot_train_accuracy', 'topt_test_accuracy',
                                'default_time', 'tpot_time']
    result_table.float_format = '.2'

    reg_datasets = pmlb.regression_dataset_names
    for ds in reg_datasets:
        print(f'Working on {ds}, [{i}/{len(reg_datasets)}]')
        row = []
        X, y = pmlb.fetch_data(ds, return_X_y=True, local_cache_dir='/home/vinayd/nwork/pmlb_cache/')
        row.append(ds)
        row.append('regression')
        row.append(X.shape[0])
        row.append(X.shape[1])

        test_size = 0.2
        row.append(test_size)

        random_state = int(random.random() * 4294967295)
        row.append(random_state)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        default_rf = RandomForestRegressor(n_estimators=100, random_state=random_state)
        tpot_rf = TPOTRegressor(template='RandomForestRegressor', generations=generations, population_size=population,
                                    verbosity=2, random_state=random_state, n_jobs=-1, max_time_mins=max_time_mins)
        start_time = time.time()
        default_rf.fit(X_train, y_train)
        end_time = time.time()
        default_time = end_time - start_time

        start_time = time.time()
        tpot_rf.fit(X_train, y_train)
        end_time = time.time()
        tpot_time = end_time - start_time

        #pdb.set_trace()
        tpot_rf_params = tpot_rf.fitted_pipeline_[0].get_params()

        default_train_accuracy = default_rf.score(X_train, y_train)
        default_test_accuracy = default_rf.score(X_test, y_test)

        tpot_train_accuracy = tpot_rf.score(X_train, y_train)
        tpot_test_accuracy = tpot_rf.score(X_test, y_test)
       
        row.append(default_train_accuracy)
        row.append(default_test_accuracy)
        row.append(tpot_train_accuracy)
        row.append(tpot_test_accuracy)

        row.append(default_time)
        row.append(tpot_time)
        result_table.add_row(row)
        #print(f'Training time = {end_time - start_time}')
        #break
        with open('regression_result.csv', 'w') as fp:
            fp.write(result_table.get_csv_string())
        i += 1


run_clf_experiments()
run_reg_experiments()
