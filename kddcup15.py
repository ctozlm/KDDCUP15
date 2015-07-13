#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'zheliu'

import re
import csv
import datetime
import math
import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn import cross_validation
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

import graphlab as gl

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.nonlinearities import softmax
from lasagne.updates import adagrad
from nolearn.lasagne import NeuralNet

dir = "/Users/zheliu/Documents/Kaggle/KDD15"


# preprocessing
def create_detail(log_pd, object_pd):
    log_pd.time = log_pd.time.apply(lambda x: re.sub("T\d+:\d+:\d+", "", x))
    log_pd.time = log_pd.time.apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d").date())
    log_pd.loc[log_pd.event == 'nagivate', 'event'] = 'navigate'
    log_pd = log_pd.groupby('course_id').apply(course_start_end_dt)
    log_pd = log_pd.groupby('username').apply(course_took)
    detail_pd = log_pd.merge(object_pd, how='left', left_on=['course_id', 'object'], right_on=['course_id', 'module_id'])
    return detail_pd


# feature engineering
def course_start_end_dt(x):
    x['course_start_dt'] = x.time.min()
    x['course_end_dt'] = x.time.max()
    return x

def course_took(x):
    x['n_course'] = len(set(x.course_id))
    x['n_course_simu'] = len(set(x.course_start_dt))
    x['user_source_ratio'] = math.log((0.5+sum(x.source=='server'))/(1+sum(x.source=='server')+sum(x.source=='browser')))
    x['user_event'] = len(x.event)
    x['user_video'] = sum(x.event=='video')
    x['user_access'] = sum(x.event=='access')
    x['user_problem'] = sum(x.event=='problem')
    return x

def create_summary(detail_pd):
    detail_pd_group = detail_pd.groupby(['enrollment_id', 'username', 'course_id'])
    n_videos = detail_pd_group['event'].apply(lambda x: sum(x=='video')).reset_index()
    n_navigate = detail_pd_group['event'].apply(lambda x: sum(x=='navigate')).reset_index()
    n_access = detail_pd_group['event'].apply(lambda x: sum(x=='access')).reset_index()
    n_problem = detail_pd_group['event'].apply(lambda x: sum(x=='problem')).reset_index()
    n_page_close = detail_pd_group['event'].apply(lambda x: sum(x=='page_close')).reset_index()
    n_discussion = detail_pd_group['event'].apply(lambda x: sum(x=='discussion')).reset_index()
    n_wiki = detail_pd_group['event'].apply(lambda x: sum(x=='wiki')).reset_index()
    n_uniq_dt = detail_pd_group['time'].apply(lambda x: len(set(x))).reset_index()
    source_ratio = detail_pd_group['source'].apply(lambda x: math.log((0.5+sum(x=='server'))/(1+sum(x=='server')+sum(x=='browser')))).reset_index()
    start_dt_access = detail_pd_group.apply(lambda x: (x['time'].min() - x['course_start_dt'].min()).days).reset_index()
    end_dt_access = detail_pd_group.apply(lambda x: (x['course_end_dt'].max() - x['time'].max()).days).reset_index()

    n_course_took = detail_pd_group['n_course'].agg(np.mean).reset_index()
    n_course_simu_took = detail_pd_group['n_course_simu'].agg(np.mean).reset_index()

    user_source_ratio = detail_pd_group['user_source_ratio'].agg(np.mean).reset_index()
    user_event = detail_pd_group.apply(lambda x: (0.5+len(x['event']))/(0.5+np.mean(x['user_event']))).reset_index()
    user_video = detail_pd_group.apply(lambda x: (0.5+sum(x['event']=='video'))/(0.5+np.mean(x['user_video']))).reset_index()
    user_access = detail_pd_group.apply(lambda x: (0.5+sum(x['event']=='access'))/(0.5+np.mean(x['user_access']))).reset_index()
    user_problem = detail_pd_group.apply(lambda x: (0.5+sum(x['event']=='problem'))/(0.5+np.mean(x['user_problem']))).reset_index()
    uniq_event = detail_pd_group.apply(lambda x: len(set(x.object))).reset_index()
    uniq_problem = detail_pd_group.apply(lambda x: len(set(x.object[x['event']=='problem']))).reset_index()
    uniq_video = detail_pd_group.apply(lambda x: len(set(x.object[x['event']=='video']))).reset_index()
    uniq_access = detail_pd_group.apply(lambda x: len(set(x.object[x['event']=='access']))).reset_index()

    n_event_0 = detail_pd_group.apply(lambda x: sum(x['time'] < (x['course_start_dt'].min()+datetime.timedelta(days=7)))).reset_index()
    n_video_0 = detail_pd_group.apply(lambda x: sum((x['time'] < (x['course_start_dt'].min()+datetime.timedelta(days=7))) & (x['event']=='video'))).reset_index()
    n_access_0 = detail_pd_group.apply(lambda x: sum((x['time'] < (x['course_start_dt'].min()+datetime.timedelta(days=7))) & (x['event']=='access'))).reset_index()
    n_problem_0 = detail_pd_group.apply(lambda x: sum((x['time'] < (x['course_start_dt'].min()+datetime.timedelta(days=7))) & (x['event']=='problem'))).reset_index()

    n_event_1 = detail_pd_group.apply(lambda x: sum(x['time'] >= (x['course_end_dt'].max()-datetime.timedelta(days=5)))).reset_index()
    n_event_2 = detail_pd_group.apply(lambda x: sum(x['time'] >= (x['course_end_dt'].max()-datetime.timedelta(days=10)))).reset_index()
    n_event_3 = detail_pd_group.apply(lambda x: sum(x['time'] >= (x['course_end_dt'].max()-datetime.timedelta(days=15)))).reset_index()
    n_video_1 = detail_pd_group.apply(lambda x: sum((x['time'] >= (x['course_end_dt'].max()-datetime.timedelta(days=5))) & (x['event']=='video'))).reset_index()
    n_video_2 = detail_pd_group.apply(lambda x: sum((x['time'] >= (x['course_end_dt'].max()-datetime.timedelta(days=10))) & (x['event']=='video'))).reset_index()
    n_video_3 = detail_pd_group.apply(lambda x: sum((x['time'] >= (x['course_end_dt'].max()-datetime.timedelta(days=15))) & (x['event']=='video'))).reset_index()
    n_navigate_1 = detail_pd_group.apply(lambda x: sum((x['time'] >= (x['course_end_dt'].max()-datetime.timedelta(days=5))) & (x['event']=='navigate'))).reset_index()
    n_navigate_2 = detail_pd_group.apply(lambda x: sum((x['time'] >= (x['course_end_dt'].max()-datetime.timedelta(days=10))) & (x['event']=='navigate'))).reset_index()
    n_navigate_3 = detail_pd_group.apply(lambda x: sum((x['time'] >= (x['course_end_dt'].max()-datetime.timedelta(days=15))) & (x['event']=='navigate'))).reset_index()
    n_access_1 = detail_pd_group.apply(lambda x: sum((x['time'] >= (x['course_end_dt'].max()-datetime.timedelta(days=5))) & (x['event']=='access'))).reset_index()
    n_access_2 = detail_pd_group.apply(lambda x: sum((x['time'] >= (x['course_end_dt'].max()-datetime.timedelta(days=10))) & (x['event']=='access'))).reset_index()
    n_access_3 = detail_pd_group.apply(lambda x: sum((x['time'] >= (x['course_end_dt'].max()-datetime.timedelta(days=15))) & (x['event']=='access'))).reset_index()
    n_problem_1 = detail_pd_group.apply(lambda x: sum((x['time'] >= (x['course_end_dt'].max()-datetime.timedelta(days=5))) & (x['event']=='problem'))).reset_index()
    n_problem_2 = detail_pd_group.apply(lambda x: sum((x['time'] >= (x['course_end_dt'].max()-datetime.timedelta(days=10))) & (x['event']=='problem'))).reset_index()
    n_problem_3 = detail_pd_group.apply(lambda x: sum((x['time'] >= (x['course_end_dt'].max()-datetime.timedelta(days=15))) & (x['event']=='problem'))).reset_index()
    n_page_close_1 = detail_pd_group.apply(lambda x: sum((x['time'] >= (x['course_end_dt'].max()-datetime.timedelta(days=5))) & (x['event']=='page_close'))).reset_index()
    n_page_close_2 = detail_pd_group.apply(lambda x: sum((x['time'] >= (x['course_end_dt'].max()-datetime.timedelta(days=10))) & (x['event']=='page_close'))).reset_index()
    n_page_close_3 = detail_pd_group.apply(lambda x: sum((x['time'] >= (x['course_end_dt'].max()-datetime.timedelta(days=15))) & (x['event']=='page_close'))).reset_index()
    n_discussion_1 = detail_pd_group.apply(lambda x: sum((x['time'] >= (x['course_end_dt'].max()-datetime.timedelta(days=5))) & (x['event']=='discussion'))).reset_index()
    n_discussion_2 = detail_pd_group.apply(lambda x: sum((x['time'] >= (x['course_end_dt'].max()-datetime.timedelta(days=10))) & (x['event']=='discussion'))).reset_index()
    n_discussion_3 = detail_pd_group.apply(lambda x: sum((x['time'] >= (x['course_end_dt'].max()-datetime.timedelta(days=15))) & (x['event']=='discussion'))).reset_index()
    n_wiki_1 = detail_pd_group.apply(lambda x: sum((x['time'] >= (x['course_end_dt'].max()-datetime.timedelta(days=5))) & (x['event']=='wiki'))).reset_index()
    n_wiki_2 = detail_pd_group.apply(lambda x: sum((x['time'] >= (x['course_end_dt'].max()-datetime.timedelta(days=10))) & (x['event']=='wiki'))).reset_index()
    n_wiki_3 = detail_pd_group.apply(lambda x: sum((x['time'] >= (x['course_end_dt'].max()-datetime.timedelta(days=15))) & (x['event']=='wiki'))).reset_index()
    
    summary_pd = pd.DataFrame({
                              'enrollment_id': n_videos['enrollment_id'],
                              'username': n_videos['username'],
                              'course_id': n_videos['course_id'],
                              'n_videos': n_videos['event'],
                              'n_navigate': n_navigate['event'],
                              'n_access': n_access['event'],
                              'n_problem': n_problem['event'],
                              'n_page_close': n_page_close['event'],
                              'n_discussion': n_discussion['event'],
                              'n_wiki': n_wiki['event'],
                              'n_uniq_dt': n_uniq_dt['time'],
                              'source_ratio': source_ratio['source'],
                              'start_dt_access': start_dt_access[0],
                              'end_dt_access': end_dt_access[0],
                              'n_course_took': n_course_took.n_course,
                              'n_course_simu_took': n_course_simu_took.n_course_simu,
                              'user_source_ratio': user_source_ratio.user_source_ratio,
                              'user_event': user_event[0],
                              'user_video': user_video[0],
                              'user_access': user_access[0],
                              'user_problem': user_problem[0],
                              'uniq_event': uniq_event[0],
                              'uniq_problem': uniq_problem[0],
                              'uniq_video': uniq_video[0],
                              'uniq_access': uniq_access[0],
                              'n_event_0': n_event_0[0],
                              'n_video_0': n_video_0[0],
                              'n_access_0': n_access_0[0],
                              'n_problem_0': n_problem_0[0],
                              'n_event_1': n_event_1[0],
                              'n_event_2': n_event_2[0],
                              'n_event_3': n_event_3[0],
                              'n_video_1': n_video_1[0],
                              'n_video_2': n_video_2[0],
                              'n_video_3': n_video_3[0],
                              'n_navigate_1': n_navigate_1[0],
                              'n_navigate_2': n_navigate_2[0],
                              'n_navigate_3': n_navigate_3[0],
                              'n_access_1': n_access_1[0],
                              'n_access_2': n_access_2[0],
                              'n_access_3': n_access_3[0],
                              'n_problem_1': n_problem_1[0],
                              'n_problem_2': n_problem_2[0],
                              'n_problem_3': n_problem_3[0],
                              'n_page_close_1': n_page_close_1[0],
                              'n_page_close_2': n_page_close_2[0],
                              'n_page_close_3': n_page_close_3[0],
                              'n_problem_1': n_problem_1[0],
                              'n_problem_2': n_problem_2[0],
                              'n_problem_3': n_problem_3[0],
                              'n_wiki_1': n_wiki_1[0],
                              'n_wiki_2': n_wiki_2[0],
                              'n_wiki_3': n_wiki_3[0]
                              })
    summary_pd = summary_pd.fillna(0)
    return summary_pd


# random forest
def random_forest(x_train, y_train):
    x_train_f, x_train_c, y_train_f, y_train_c = train_test_split(x_train, y_train, test_size = 0.1)
    for k in range(5, 30):
        clf = RandomForestClassifier(n_estimators = 200, max_features = k)
        cv = cross_validation.KFold(len(x_train_f), n_folds = 5, shuffle = True)
        results = []
        for traincv, testcv in cv:
            probas = clf.fit(x_train_f[traincv], y_train_f[traincv]).predict_proba(x_train_f[testcv])
            fpr, tpr, thresholds = metrics.roc_curve(y_train_f[testcv], probas[:,1])
            results.append(metrics.auc(fpr, tpr))
        print "Results: " + str(k) + " " + str(np.array(results).mean())
    
    k = 11
    clf = RandomForestClassifier(n_estimators = 200, criterion = "gini", max_features = k)
    clf.fit(x_train_f, y_train_f)
    calib_clf = CalibratedClassifierCV(clf, method = "isotonic", cv = "prefit")
    calib_clf.fit(x_train_c, np.array(y_train_c).astype(int))
    return calib_clf


# gradient boosting
def gradient_boosting(x_train, y_train):
    dat = np.concatenate([x_train, np.array([y_train]).T], axis=1)
    pd.dat = pd.DataFrame(dat)
    ind = dat.shape[1] - 1
    pd.dat[ind] = pd.dat[ind].astype(int)
    train = gl.SFrame(pd.dat)
    
    params = {'target': str(ind),
              'max_iterations': 500,
              'step_size': 0.04,
              'max_depth': 5,
              'min_child_weight': 4,
              'row_subsample': .9,
              'min_loss_reduction': 1,
              'column_subsample': .8,
              'validation_set': None}

    gbm = gl.boosted_trees_classifier.create(train, **params)
    return gbm


# neural network
def load_train_data(x_train, y_train):
    dat = np.concatenate([x_train, np.array([y_train]).T], axis=1)
    df = pd.DataFrame(dat)
    X = df.values.copy()
    np.random.shuffle(X)
    X, labels = X[:,0:-1].astype(np.float32), X[:,-1]
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels).astype(np.int32)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y, encoder, scaler

def load_test_data(x_test, scaler):
    df = pd.DataFrame(x_test)
    X = df.values.copy()
    X = X[:, 0:].astype(np.float32)
    X = scaler.transform(X)
    return X

def neural_network(x_train, y_train):
    X, y, encoder, scaler = load_train_data(x_train, y_train)
    num_classes = len(encoder.classes_)
    num_features = X.shape[1]
    layers0 = [('input', InputLayer),
               ('dropoutf', DropoutLayer),
               ('dense0', DenseLayer),
               ('dropout', DropoutLayer),
               ('dense1', DenseLayer),
               ('dropout2', DropoutLayer),
               ('output', DenseLayer)]
    net0 = NeuralNet(layers=layers0,
                input_shape=(None, num_features),
                dropoutf_p=0.15,
                dense0_num_units=1000,
                dropout_p=0.25,
                dense1_num_units=500,
                dropout2_p=0.25,
                output_num_units=num_classes,
                output_nonlinearity=softmax,
                update=adagrad,
                update_learning_rate=0.005,
                eval_size=0.01,
                verbose=1,
                max_epochs=30)
    net0.fit(X, y)
    return (net0, scaler)


# ensemble
def loss_func(weights):
    final_prediction = 0
    for weight, prediction in zip(weights, predictions):
        final_prediction += weight * np.array(prediction)
    fpr, tpr, thresholds = metrics.roc_curve(y_valid, final_prediction)
    res = metrics.auc(fpr, tpr)
    return res


def main():
    # load train data
    label_pd = pd.read_csv(dir + '/train/truth_train.csv', header=None, sep=',')
    label_pd.columns = ['enrollment_id', 'label']
    enroll_pd = pd.read_csv(dir + '/train/enrollment_train.csv', sep=',')
    object_pd = pd.read_csv(dir + '/object.csv', sep=',')
    enroll_pd = enroll_pd[enroll_pd.enrollment_id != 139669]
    enroll_pd = enroll_pd.merge(label_pd, on=['enrollment_id'])
    log_pd = pd.read_csv(dir + '/train/log_train.csv', sep=',')
    log_pd = log_pd[log_pd.enrollment_id != 139669]
    detail_pd = create_detail(log_pd, object_pd)
    summary_pd = create_summary(detail_pd)
    summary_pd = summary_pd.merge(label_pd, on=['enrollment_id'])

    # load test data
    enroll_test_pd = pd.read_csv(dir + '/test/enrollment_test.csv', sep=',')
    log_test_pd = pd.read_csv(dir + '/test/log_test.csv', sep=',')
    detail_test_pd = create_detail(log_test_pd, object_pd)
    summary_test_pd = create_summary(detail_test_pd)

    # transformation
    course_dummies = pd.get_dummies(summary_pd['course_id'], prefix='course_id')
    xtrain = summary_pd.drop(['course_id','enrollment_id', 'username', 'label'], axis=1).join(course_dummies)
    xtrain = np.array(xtrain)
    ytrain = np.array(summary_pd['label'])
    course_dummies_test = pd.get_dummies(summary_test_pd['course_id'], prefix='course_id')
    xtest = summary_test_pd.drop(['course_id','enrollment_id', 'username'], axis=1).join(course_dummies_test)
    xtest = np.array(xtest)

    # split into train + valid
    x_train, x_valid, y_train, y_valid = train_test_split(xtrain, ytrain, test_size = 0.1)

    # random forest
    calib_clf = random_forest(x_train, y_train)
    fpr, tpr, thresholds = metrics.roc_curve(y_valid, calib_clf.predict_proba(x_valid)[:,1])
    print "RF" + " " + str(metrics.auc(fpr, tpr))

    # gradient boosting
    gbm = gradient_boosting(x_train, y_train)
    x_valid_gbm = gl.SFrame(pd.DataFrame(x_valid))
    preds = gbm.predict_topk(x_valid_gbm, output_type = 'probability', k = 2)
    preds = preds.unstack(['class', 'probability'], 'probs').unpack('probs', '')
    preds['id'] = preds['id'].astype(int) + 1
    preds = preds.sort('id')
    del preds['id']
    preds_df = preds.to_dataframe()
    fpr, tpr, thresholds = metrics.roc_curve(y_valid, [p[1] for p in preds_df.values])
    print "GBM" + " " + str(metrics.auc(fpr, tpr))

    # fit neural network
    net0, scaler = neural_network(x_train, y_train)
    x_valid_nn = load_test_data(x_valid, scaler)
    fpr, tpr, thresholds = metrics.roc_curve(y_valid, [p[1] for p in net0.predict_proba(x_valid_nn)])
    print "NN" + " " + str(metrics.auc(fpr, tpr))
    
    # ensemble
    p1 = calib_clf.predict_proba(x_valid)[:,1]
    p2 = [p[1] for p in preds_df.values]
    p3 = [p[1] for p in net0.predict_proba(x_valid_nn)]
    predictions = []
    predictions.append(p1)
    predictions.append(p2)
    predictions.append(p3)
    local_max = 0
    local_weight = [0.1,0.5,0.4]
    for w1 in np.arange(0,1,0.01):
        for w2 in np.arange(0,1-w1,0.01):
                tmp = loss_func([w1, w2, 1-w1-w2])
                if tmp > local_max:
                    local_max = tmp
                    local_weight = [w1, w2, 1-w1-w2]

    # predict
    preds_rf = calib_clf.predict_proba(xtest)[:,1].tolist()

    x_test_gbm = gl.SFrame(pd.DataFrame(xtest))
    preds = gbm.predict_topk(x_test_gbm, output_type = 'probability', k = 2)
    preds = preds.unstack(['class', 'probability'], 'probs').unpack('probs', '')
    preds['id'] = preds['id'].astype(int) + 1
    preds = preds.sort('id')
    del preds['id']
    preds_df = preds.to_dataframe()
    preds_gbm = [p[1] for p in preds_df.values]
    
    x_test_nn = load_test_data(xtest, scaler)
    preds_nn_prob = net0.predict_proba(x_test_nn)
    preds_nn = [p[1] for p in preds_nn_prob]
    
    # submit
    preds_ensemble = np.array(preds_rf) * local_weight[0] + np.array(preds_gbm) * local_weight[1] + np.array(preds_nn) * local_weight[2]
    output = pd.DataFrame({'enrollment_id': enroll_test_pd.enrollment_id,'prob': preds_ensemble})
    output.to_csv(dir + '/submission.csv', header=False, index=False, sep=',')


if __name__ == "__main__":
    main()