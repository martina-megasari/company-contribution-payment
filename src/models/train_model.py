import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
import statsmodels.api as sm

import lightgbm as lgb
from hyperopt import hp, tpe, Trials
from hyperopt.fmin import fmin

import math
import copy
import pickle
from datetime import datetime
from setup import CONFIG, ROOT_DIR
import os
# import shap

from sklearn.metrics import mean_squared_error, mean_absolute_error
import logging
from src.features.build_features import build_features
from sklearn.model_selection import KFold

pd.options.mode.chained_assignment = None
logging.basicConfig(level=logging.INFO)

max_round = 500
early_stopping = 125
err_metric = 'rmse'
k_cv = 5
max_hyperopt_eval = 100

l_choice_params = ['num_leaves', 'subsample_for_bin', 'boosting_type', 'min_data_in_leaf', 'lambda_l1', 'lambda_l2', 'max_depth']
choice = {}
choice['num_leaves'] = np.arange(30, 150, dtype=int)
choice['subsample_for_bin'] = np.arange(200, 3000, 200, dtype=int)
choice['lambda_l1'] = [0, hp.loguniform('lambda_l1_positive', -16, 2)]
choice['lambda_l2'] = [0, hp.loguniform('lambda_l2_positive', -16, 2)]
choice['boosting_type'] = ['gbdt', 'goss']
choice['min_data_in_leaf'] = np.arange(8, 100, dtype=int)
choice['max_depth'] = np.arange(5, 20, dtype=int)


hyperopt_space = {
        'boosting_type': hp.choice('boosting_type', choice['boosting_type']),
        'num_leaves':  hp.choice('num_leaves', choice['num_leaves']),
        'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.2)),
        'subsample_for_bin': hp.choice('subsample_for_bin', choice['subsample_for_bin']),
        'feature_fraction': hp.uniform('feature_fraction', 0.5, 1),
        'bagging_fraction': hp.uniform('bagging_fraction', 0.5, 1), #alias "subsample"
        'min_data_in_leaf': hp.choice('min_data_in_leaf', choice['min_data_in_leaf']),
        'max_depth': hp.choice('max_depth', choice['max_depth']),
        'lambda_l1': hp.choice('lambda_l1',choice['lambda_l1']),
        'lambda_l2': hp.choice('lambda_l2', choice['lambda_l2'])
    }


class PredictionModel:
    def __init__(self, model, algorithm):
        self.model = model
        self.algo = algorithm


def get_base_regressor(params) -> lgb.sklearn.LGBMRegressor:
    """
    Create the LGBMRegressor object to train
    :param params: hyperparameters to build the model
    :return: LGBMRegressor
    """
    regressor = lgb.sklearn.LGBMRegressor(
        n_estimators=max_round,
        metrics=err_metric,
        early_stopping_round=early_stopping,
        verbose=-1,
        **params
    )
    return regressor


def objective(params):
    """
    Objective function to be optimized by hyperopt, perform k-fold cross validation on the model
    :param params: hyperparameters to be tested
    :return: the mean of the cross validation score
    """
    l_score = []
    cv = KFold(random_state=CONFIG['random_state'], shuffle=True)
    for fold_, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
        regressor = get_base_regressor(params)
        X_train_cv, y_train_cv = X_train.iloc[train_idx], y_train.iloc[train_idx]
        X_val_cv, y_val_cv = X_train.iloc[val_idx], y_train.iloc[val_idx]
        regressor.fit(X_train_cv, y_train_cv, eval_set=[(X_val_cv, y_val_cv)], eval_metric=err_metric, verbose=False)
        y_val_pred = regressor.predict(X_val_cv)
        l_score.append(math.sqrt(mean_squared_error(y_val_cv, y_val_pred)))

    return np.mean(l_score)


def build_lightgbm(x_train, y_train) -> lgb.sklearn.LGBMRegressor:
    """
    Tune the hyperparameters and use the best combinations to build a light gbm model
    :param x_train: training features
    :param y_train: traning label
    :return: trained LGBMRegressor
    """
    trial = Trials()
    best = fmin(fn=objective,
                algo=tpe.suggest,
                max_evals=max_hyperopt_eval,
                trials=trial,
                space=hyperopt_space
                )

    best_par = copy.deepcopy(best)
    for param in l_choice_params:
        if param in ['lambda_l1', 'lambda_l2'] and best_par[param] == 1:
            best_par[param] = best_par['{}_{}'.format(param, 'positive')]
        else:
            best_par[param] = choice[param][best_par[param]]

    x_t, x_val, y_t, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=46)
    gbm = get_base_regressor(best_par)
    gbm.fit(x_t, y_t, eval_set=[(x_val, y_val)], eval_metric=err_metric)
    return gbm


def build_model(X_train:pd.DataFrame, y_train, algo="linreg") -> PredictionModel:
    """
    Build a regression model
    :param X_train: training features
    :param y_train: training label
    :param algo: the model algorithm
    :return: PredictionModel object that contains the model itself along with some information around it
    """
    if algo == "linreg":
        X_train = X_train.to_numpy()
        X2 = sm.add_constant(X_train)
        est = sm.OLS(y_train, X2)
        est2 = est.fit()
        logging.info("Linear regression model has been built succesfully!")
        logging.info(est2.summary())
        return PredictionModel(est2, algo)
    elif algo == "lightgbm":
        regressor = build_lightgbm(X_train, y_train)
        return PredictionModel(regressor, algo)
    else:
        logging.error("Failed to build a prediction model, algo param is not recognized")
        return None


def make_prediction(model: PredictionModel, X_test):
    """
    Use the model to make prediction
    :param model: PredictionModel object
    :param X_test: test features
    :return: an array of predicted value
    """
    y_test_pred = None
    if model.algo == "linreg":
        X2_test = sm.add_constant(X_test)
        y_test_pred = model.model.predict(X2_test)
    elif model.algo == "lightgbm":
        y_test_pred = model.model.predict(X_test)
    else:
        logging.error("Failed to make prediction, algo param is not recognized")
    return y_test_pred


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = build_features()
    target = CONFIG['prediction_target']
    algo = CONFIG['algorithm']

    model = build_model(X_train, y_train, algo)
    y_test_pred = make_prediction(model, X_test)

    if target == 'contribution':
        logging.info("Prediction performance")
        logging.info("MAE: {}".format(mean_absolute_error(y_test_pred, y_test)))
        logging.info("RMSE: {}".format(math.sqrt(mean_squared_error(y_test_pred, y_test))))
    else:
        logging.info("Prediction performance on the log_contribution")
        logging.info("MAE: {}".format(mean_absolute_error(y_test_pred, y_test)))
        logging.info("RMSE: {}".format(math.sqrt(mean_squared_error(y_test_pred, y_test))))

        logging.info("Prediction performance on the contribution")
        logging.info("MAE: {}".format(mean_absolute_error(np.exp(y_test_pred), np.exp(y_test))))
        logging.info("RMSE: {}".format(math.sqrt(mean_squared_error(np.exp(y_test_pred), np.exp(y_test)))))

    save_to = os.path.join(ROOT_DIR, 'models/{}_{}.pkl'.format(algo, datetime.now().strftime("%Y%m%d_%H%M%S")))
    pickle.dump(model, open(save_to, "wb"))

    # sns.regplot(x=y_test_pred, y=y_test.values)
    # plt.xlabel("prediction")
    # plt.ylabel("actual")
    # plt.title("Actual vs prediction ")
    # plt.show()