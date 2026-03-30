
import os
import pandas as pd
import numpy as np
import tensorflow as tf

import dask.dataframe as ddf
from dask.diagnostics import ProgressBar

from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GroupKFold

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from datetime import datetime 
from pytz import timezone

class Runner(object):
    """Wrapper of model.ModelNN_tensorflow class to run hold-out training and cross validation.

    Args:
        object ([type]): [description]
    """    
    def __init__(self, model_cls, network_structure_cls, model_cls_params={}, run_name=None):
        """Instanciate runner object with specified model class and model parameters.

        Args:
            model_cls ([abc.ABCMeta]): Model class inherited from MyAbstractModel.
            network_structure_cls ([abc.ABCMeta]):  NetworkStructure class inherited from MyAbstractNetworkStructure. 
                This arg determines the structure of the neural network.
            model_cls_params (dict, optional): Model parameters. Accepted arguments depend on network_cls. Defaults to {}.
            run_name ([str], optional): Name of the run. If None is given, run_name is set to 'runner_yyyymmdd_hhmm'.  Defaults to None.

        Example usage:
            >>> rn = runner.Runner(
                    model_cls=ModelNN_tensorflow, 
                    network_structure_cls=ElementwiseDNN4Tuning, 
                    model_cls_params={'lr': 1e-4}
                )
            >>> cv_result = rn.run_train_cv(X, y, n_fold=4, **fit_params)
        
        Returns:
            [type]: [description]
        """        
        self.model_cls = model_cls
        self.network_structure_cls = network_structure_cls
        self.model_cls_params = model_cls_params
        
        self.paths = {}
        if run_name is None:
            run_name = 'runner_' + datetime.now(timezone('Asia/Tokyo')).strftime('%Y%m%d_%H%M')
        self.run_name = run_name
        return None

    def _run_train_holdout_pandas(self, X, y, X_valid, y_valid, **fit_params):
        """Run hold-out training by using X, y of data type pandas.DataFrame.
        This function runs model.fit and model.evaluate and returns result dict.

        Args:
            X ([pandas.core.frame.DataFrame]): Input data.
            y ([pandas.core.frame.DataFrame]): Target data.
            X_valid ([pandas.core.frame.DataFrame]): Input data for validation.
            y_valid ([pandas.core.frame.DataFrame]): Target data for validation.

        Returns:
            result [dict]: result dictionary that contains model, train result and validation result.
                train result and validation result respectively contains 'score', 'index' and 'prediction'.
        """        
        # instanciate model ----------------------------------------------------
        model = self.model_cls(network_cls=self.network_structure_cls, **self.model_cls_params)

        # split related functions were removed. 
        # split related functions were moved to run_train_cv functions

        # train the model ---------------------------------------------------
        model.fit(X=X, y=y, validation_data=(X_valid, y_valid), **fit_params)

        train_score, train_preds = model.evaluate(X=X, y=y, **fit_params)
        valid_score, valid_preds = model.evaluate(X=X_valid, y=y_valid, **fit_params)

        # make index ---------------------------------------------------
        train_idx = train_preds.index.values # .tolist() 
        valid_idx = valid_preds.index.values # .tolist()

        # summarize the result ---------------------------------------------------
        result = {
            'model': model,
            'validation':{
                'score': valid_score,
                'index': valid_idx,
                'prediction': valid_preds},
            'train':{
                'score': train_score,
                'index': train_idx,
                'prediction': train_preds},
        }
        return result


    def _run_train_holdout_dataset(self, train_dataset, valid_dataset, **fit_params):     
        """Run hold-out training by using train_dataset and valid_dataset of data TFRecordDatasetV2.
        This function runs model.fit and model.evaluate and returns result dict.

        Args:
            train_dataset ([TFRecordDatasetV2]): Input and target data.
            valid_dataset ([TFRecordDatasetV2]): Input and target data for validation.

        Returns:
            result [dict]: result dictionary that contains model, train result and validation result.
                train result and validation result respectively contains 'score', 'index' and 'prediction'.
        """ 
        # instanciate model ----------------------------------------------------
        model = self.model_cls(network_cls=self.network_structure_cls, **self.model_cls_params)

        # train the model ---------------------------------------------------
        model.fit(train_dataset, validation_data=valid_dataset, **fit_params)

        train_score, train_preds = model.evaluate(train_dataset, **fit_params)
        valid_score, valid_preds = model.evaluate(valid_dataset, **fit_params)

        # TODO: support indexing by tfrecord way ---------------------------------------------------
        # This is a temporary measure
        # make tentative index
        train_idx = train_preds.index.values #.tolist() # integer index 0~len(train_preds)
        valid_idx = valid_preds.index.values #.tolist() # integer index 0~len(train_preds)
        # train_idx = model.network_structure.generate_index(train_dataset)
        # valid_idx = model.network_structure.generate_index(valid_dataset)

        # generate label ---------------------------------------------------
        # train_label = model.network_structure.preprocess(train_dataset).map(lambda feature, label: label).batch(1028)
        # train_label = np.concatenate([t.numpy() for t in train_label])
        # train_label = model.network_structure.shape_preds(train_label, index=None)
        train_label = model.network_structure.generate_label(train_dataset)

        
        # valid_label = model.network_structure.preprocess(valid_dataset).map(lambda feature, label: label).batch(1028)
        # valid_label = np.concatenate([t.numpy() for t in valid_label])
        # valid_label = model.network_structure.shape_preds(valid_label, index=None)
        valid_label = model.network_structure.generate_label(valid_dataset)

        # summarize the result ---------------------------------------------------
        result = {
            'model': model,
            'validation':{
                'score': valid_score,
                'index': valid_idx,
                'prediction': valid_preds,
                'label': valid_label},
            'train':{
                'score': train_score,
                'index': train_idx,
                'prediction': train_preds,
                'label': train_label},
        }
        return result

    def run_train_holdout(self, X, y=None, validation_data=None, **fit_params):
        """ holod-out training function that executes _run_train_holdout_pandas or _run_train_holdout_dataset depending on the data type

        Args:
            X : Input data. Pandas DataFrame or TFRecordDatasetV2 can be accepted.
            y ([pandas.DataFram], optional): Target data. If the type of X is TFRecordDatasetV2, please set this to None. Defaults to None.
            validation_data (optional): Input and Target data for validation. 
                tuple of Pandas DataFrame (example: tuple(X_valid, y_valid)) or TFRecordDatasetV2 can be accepted. Defaults to None.

        Raises:
            TypeError: Raised if the types of X, y and validation_data is not  pandas DataFrame or TFRecordDatasetV2.

        Returns:
            result [dict]: result dictionary that contains model, train result and validation result.
                train result and validation result respectively contains 'score', 'index' and 'prediction'.

        """        
        if isinstance(X, tf.python.data.ops.readers.TFRecordDatasetV2) and y is None:
            result = self._run_train_holdout_dataset(X, validation_data, **fit_params)
        elif isinstance(X, pd.core.frame.DataFrame) and isinstance(y, pd.core.frame.DataFrame):
            X_valid, y_valid = validation_data
            result = self._run_train_holdout_pandas(X, y, X_valid, y_valid, **fit_params)
        else:
            raise TypeError('''Feature(X) and label(y) must be pandas.DataFrame or TFRecordDataset.
            Please check the type of your data.
            CAUTION: If X is TFRecordDataset, y must be None.''')
        return result

    def _run_train_cv_pandas(self, X, y, n_fold=3, save_models=True, output_dir='./', **fit_params):
        """Run K-fold cross validation training by using X, y of data type pandas.DataFrame.
        This function divides the input data into n_fold pieces (K_1,...,K_n) and executes run_train_holdout n_fold times.
        The i-th run of run_train_holdout is executed with K_i-th splited data as the validation data and the rest as the training data.

        Args:
            X ([pandas.core.frame.DataFrame]): Input data.
            y ([pandas.core.frame.DataFrame]): Target data.
            n_fold (int, optional): the number of groups that a given data sample is to be split into. Defaults to 3.
            save_models (bool, optional): Whether to save the models of each fold. Defaults to True.
            output_dir (str, optional): Output directory path to save the model. Defaults to './'.

        Returns:
            scores_df [pandas.core.frame.DataFrame]: pandas.DataFrame with evaluation metrics as column names and Fold as index. The evaluation metrics is dependent on network_structure_cls.
            preds_df [pandas.core.frame.DataFrame]: pandas.DataFrame with prediction value, ground truth and fold as column names and sample id as index.
            valid_indices [list]: List of indices which is used as validation data in each fold.
            logs_df [pandas.core.frame.DataFrame]: pandas.DataFrame with evaluation metrics, epoch and fold as column names. The evaluation metrics is dependent on network_structure_cls.
        
        """        
        scores = []
        valid_indices = []
        preds = []
        logs = []

        kf = KFold(n_splits=n_fold, random_state=42, shuffle=True)
        for fold, (train_idx, valid_idx) in enumerate(kf.split(y.index)):
            # assign train and validation -----------------------------------------------------
            train_idx = y.index[train_idx]
            valid_idx = y.index[valid_idx]

            X_train, y_train = X.loc[train_idx], y.loc[train_idx]
            X_valid, y_valid = X.loc[valid_idx], y.loc[valid_idx]
            
            # Run hold-out training -----------------------------------------------------
            result = self.run_train_holdout(X=X_train, y=y_train, validation_data=(X_valid, y_valid), **fit_params)
            
            # save model and log -----------------------------------------------------
            model = result['model']
            if save_models:
                # set paths -----------------------------------------------------
                model_path = os.path.join(output_dir, self.run_name, f"DNN_model_fold_{fold:03}.h5")
                log_path = os.path.join(output_dir, self.run_name, f"log_{fold:03}.csv")
                self.paths[f"fold_{fold}"] = {'model': model_path, 'log': log_path}

                model.save_model(model_path)
                model.log.to_csv(log_path)
            
            # append to result lists -----------------------------------------------------
            scores.append(result['validation']['score'])
            valid_indices.append(result['validation']['index'])
            preds.append(result['validation']['prediction'])
            logs.append(model.log)
        
        # summarize scores and predictions ---------------------------------------------------
        scores_df = Runner.shape_cv_scores(scores)
        preds_df = Runner.shape_cv_preds(preds, y, use_index=True)
        logs_df = Runner.shape_cv_logs(logs)

        # scores_df = pd.DataFrame(
        #     scores, 
        #     index=[f"fold_{fold}" for fold in range(len(scores))]
        # )
        
        # preds_df = pd.merge(
        #     pd.concat(preds), y, 
        #     how='left', left_index=True, right_index=True, 
        #     suffixes=('_preds', '_true')
        # ).assign(
        #     fold=np.concatenate(
        #         [np.repeat(fold, len(valid_index)) 
        #             for fold, valid_index in enumerate(valid_indices)]
        #     )
        # )   
        # logs_df = pd.concat([log.assign(fold=i) for i, log in enumerate(logs)])
        return scores_df, preds_df, valid_indices, logs_df

    def _run_train_cv_dataset(self, tfrecord_files, n_fold=3, save_models=True, output_dir='./', use_index=True, **fit_params):
        """Run K-fold cross validation training by using input data of data type TFRecordDatasetV2.
        To be precise, it takes a list of TFRecordDatasetV2 file paths and splits the list to create Train data and Validation data.
        This function divides the input list into n_fold pieces (K_1,...,K_n) and create TFRecordDataset by tf.data.TFRecordDataset(tfr_files).
        Then, executes run_train_holdout by train TFRecordDataset and validation TFRecordDataset n_fold times.
        The i-th run of run_train_holdout is executed with K_i-th splited list of file paths as the validation data and the rest as the training data.


        Args:
            tfrecord_files ([list]): List of TFRecordDatasetV2 file paths of input and target data.
            n_fold (int, optional): the number of groups that a given data sample is to be split into. Defaults to 3.
            save_models (bool, optional): Whether to save the models of each fold. Defaults to True.
            output_dir (str, optional): Output directory path to save the model. Defaults to './'.

        Returns:
            scores_df [pandas.core.frame.DataFrame]: pandas.DataFrame with evaluation metrics as column names and Fold as index. The evaluation metrics is dependent on network_structure_cls.
            preds_df [pandas.core.frame.DataFrame]: pandas.DataFrame with prediction value, ground truth and fold as column names and sample id as index.
            valid_indices [list]: List of indices which is used as validation data in each fold.
            logs_df [pandas.core.frame.DataFrame]: pandas.DataFrame with evaluation metrics, epoch and fold as column names. The evaluation metrics is dependent on network_structure_cls.
        
        """        
        # result lists -----------------------------------------------------
        scores = []
        valid_indices = []
        preds = []
        labels = []
        logs = []

        kf = KFold(n_splits=n_fold, random_state=42, shuffle=True)
        for fold, (train_idx, valid_idx) in enumerate(kf.split(tfrecord_files)):
            # assign train and validation -----------------------------------------------------
            train_tfr_files = [tfrecord_files[idx] for idx in train_idx]
            valid_tfr_files = [tfrecord_files[idx] for idx in valid_idx]
            tfr_train = tf.data.TFRecordDataset(train_tfr_files)
            tfr_valid = tf.data.TFRecordDataset(valid_tfr_files)

            # run hold out training -----------------------------------------------------
            result = self.run_train_holdout(tfr_train, validation_data=tfr_valid, **fit_params)

            # save model and log -----------------------------------------------------
            model = result['model']
            if save_models:
                # set paths -----------------------------------------------------
                model_path = os.path.join(output_dir, self.run_name, f"DNN_model_fold_{fold:03}.h5")
                log_path = os.path.join(output_dir, self.run_name, f"log_{fold:03}.csv")
                self.paths[f"fold_{fold}"] = {'model': model_path, 'log': log_path}

                model.save_model(model_path)
                model.log.to_csv(log_path)
                
            # append to result lists -----------------------------------------------------
            scores.append(result['validation']['score'])
            valid_indices.append(result['validation']['index'])
            preds.append(result['validation']['prediction'])
            labels.append(result['validation']['label'])
            logs.append(model.log)
        
        # summarize scores and predictions ---------------------------------------------------
        scores_df = Runner.shape_cv_scores(scores)
        preds_df = Runner.shape_cv_preds(preds, labels, use_index=use_index)
        logs_df = Runner.shape_cv_logs(logs)

        # scores_df = pd.DataFrame(scores, index=[f"fold_{fold}" for fold in range(len(scores))])
        
        # preds_df = pd.merge(
        #     pd.concat(preds).reset_index(drop=True), pd.concat(labels).reset_index(drop=True), 
        #     how='left', left_index=True, right_index=True, 
        #     suffixes=('_preds', '_true')
        # ).assign(
        #     fold=np.concatenate(
        #         [np.repeat(fold, valid_index.shape[0]) 
        #             for fold, valid_index in enumerate(valid_indices)]
        #     )
        # )   

        # logs_df = pd.concat([log.assign(fold=i) for i, log in enumerate(logs)])
        return scores_df, preds_df, valid_indices, logs_df

    def run_train_cv(self, X, y=None, n_fold=3, save_models=True, output_dir='./', use_index=True, **fit_params):
        """cross-validation training function that executes _run_train_cv_pandas or _run_train_cv_dataset depending on the data type
        
        Train by pandas.DataFrame:
            If you wlold like to train by pandas.DataFrame, please pass pandas.DataFrame as X and y arguments.
            See _run_train_cv_pandas for details. _run_train_cv_pandas function divides the input data into n_fold pieces (K_1,...,K_n) and executes run_train_holdout n_fold times.
            The i-th run of run_train_holdout is executed with K_i-th splited data as the validation data and the rest as the training data.
        
        Train by TFRecordDatasetV2:
            If you wlold like to train by TFRecordDatasetV2, please pass a list of TFRecordDatasetV2 file paths as X argument and None as y.
            See _run_train_cv_dataset for details. _run_train_cv_dataset function divides the input list into n_fold pieces (K_1,...,K_n) and create TFRecordDataset by tf.data.TFRecordDataset(tfr_files).
            Then, executes run_train_holdout by train TFRecordDataset and validation TFRecordDataset n_fold times.
            The i-th run of run_train_holdout is executed with K_i-th splited list of file paths as the validation data and the rest as the training data.

        Args:
            X : Input data. pandas.DataFrame or a list of TFRecord file paths can be accepted.
            y ([pandas.DataFram], optional): Target data. If the type of X is TFRecordDatasetV2, please set this to None. Defaults to None.
            n_fold (int, optional): the number of groups that a given data sample is to be split into. Defaults to 3.
            save_models (bool, optional): Whether to save the models of each fold. Defaults to True.
            output_dir (str, optional): Output directory path to save the model. Defaults to './'.

        Raises:
            TypeError: Raised if the types of X, y and validation_data is not  pandas DataFrame or a list of TFRecord file paths.

        Returns:
            scores_df [pandas.core.frame.DataFrame]: pandas.DataFrame with evaluation metrics as column names and fold as index. The evaluation metrics is dependent on network_structure_cls.
            preds_df [pandas.core.frame.DataFrame]: pandas.DataFrame with prediction value, ground truth and fold as column names and sample id as index.
            valid_indices [list]: List of indices which is used as validation data in each fold.
            logs_df [pandas.core.frame.DataFrame]: pandas.DataFrame with evaluation metrics, epoch and fold as column names. The evaluation metrics is dependent on network_structure_cls.
        
        """        
        if isinstance(X, list) and y is None:
            scores_df, preds_df, valid_indices, logs_df = self._run_train_cv_dataset(X, n_fold=n_fold, save_models=save_models, output_dir=output_dir, use_index=use_index, **fit_params)
        elif isinstance(X, pd.core.frame.DataFrame) and isinstance(y, pd.core.frame.DataFrame):
            scores_df, preds_df, valid_indices, logs_df = self._run_train_cv_pandas(X, y, n_fold=n_fold, save_models=save_models, output_dir=output_dir, **fit_params)
        else:
            raise TypeError('''Feature(X) and label(y) must be pandas.DataFrame or the list of TFRecord file path.
            Please check the type of your data.
            CAUTION: If X is the list of TFRecord file paths, y must be None.''')
        return scores_df, preds_df, valid_indices, logs_df

    # previous version of groupcv which split the input by its index.
    def run_train_groupcv(self, X, y, groups, save_models=True, output_dir='./', **fit_params):
        """Not Used. previous version of groupcv which split the input by its index.

        Args:
            X ([type]): [description]
            y ([type]): [description]
            groups ([type]): [description]
            save_models (bool, optional): [description]. Defaults to True.
            output_dir (str, optional): [description]. Defaults to './'.

        Returns:
            [type]: [description]
        """        
        scores = []
        valid_indices = []
        preds = []
        logs = []
        
        n_fold = len(np.unique(groups))
        kf = GroupKFold(n_splits=n_fold)

        for fold, (train_idx, valid_idx) in enumerate(kf.split(X, y, groups)):
            # train_idx.shape, valid_idx.shape
            # print(X.iloc[train_idx].index.str.extract('smina_(.{4})_')[0].unique())
            # print(X.iloc[valid_idx].index.str.extract('smina_(.{4})_')[0].unique())

            train_idx = y.index[train_idx]
            valid_idx = y.index[valid_idx]

            result = self.__run_train_holdout_pandas(X=X, y=y, valid_idx=valid_idx, **fit_params)
            model = result['model']

            if save_models:
                model_path = os.path.join(output_dir, self.run_name, f"DNN_model_fold_{fold:03}.h5")
                log_path = os.path.join(output_dir, self.run_name, f"log_{fold:03}.csv")
                self.paths[f"fold_{fold}"] = {'model': model_path, 'log': log_path}
                model.save_model(model_path)
                model.log.to_csv(log_path)

            scores.append(result['validation']['score'])
            valid_indices.append(result['validation']['index'])
            preds.append(result['validation']['prediction'])
            logs.append(model.log)
            
        # summarize scores and predictions ---------------------------------------------------
        scores_df = pd.DataFrame(
            scores, 
            index=[f"fold_{fold}" for fold in range(len(scores))]
        )
        
        preds_df = pd.merge(
            pd.concat(preds), y, 
            how='left', left_index=True, right_index=True, 
            suffixes=('_preds', '_true')
        ).assign(
            fold=np.concatenate(
                [np.repeat(fold, valid_index.shape[0]) 
                    for fold, valid_index in enumerate(valid_indices)]
            )
        )      
        return scores_df, preds_df, valid_indices, logs # scores, preds, valid_indices, logs

    # previous version that can accept validation data as valid_idx and holds split function in it.
    def __run_train_holdout_pandas(self, X, y, valid_idx=None, test_size=0.33, stratify=None, **fit_params):
        """Not Used. previous version that can accept validation data as valid_idx and holds split function in it.

        Args:
            X ([type]): [description]
            y ([type]): [description]
            valid_idx ([type], optional): [description]. Defaults to None.
            test_size (float, optional): [description]. Defaults to 0.33.
            stratify ([type], optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """        
        # instanciate model ----------------------------------------------------
        model = self.model_cls(**self.model_cls_params)

        # split X and y into train and validation set --------------------------
        if valid_idx is None:
            train_idx, valid_idx = train_test_split(y.index, test_size=test_size, stratify=stratify)
        else:
            train_idx = y.index.values[~np.isin(y.index, valid_idx)] # get the complements of valid_idx
        X_train, y_train = X.loc[train_idx], y.loc[train_idx]
        X_valid, y_valid = X.loc[valid_idx], y.loc[valid_idx]

        # train the model ---------------------------------------------------
        model.fit(X=X_train, y=y_train, validation_data=(X_valid, y_valid), **fit_params)

        train_score, train_preds = model.evaluate(X=X_train, y=y_train, **fit_params)
        valid_score, valid_preds = model.evaluate(X=X_valid, y=y_valid, **fit_params)

        # summarize the result ---------------------------------------------------
        result = {
            'model': model,
            'validation':{
                'score': valid_score,
                'index': valid_idx,
                'prediction': valid_preds},
            'train':{
                'score': train_score,
                'index': train_idx,
                'prediction': train_preds},
        }
        return result

    @staticmethod
    def shape_cv_scores(scores):
        """Function to format the scores from run_train_cv functions.
        Summarize the score at each fold as a pandas.DataFrame.

        Args:
            scores ([list]): list of score dict.

        Returns:
            scores_df [pandas.core.frame.DataFrame]: Summarized dataframe of the score at each fold.
        """        
        scores_df = pd.DataFrame(scores, index=[f"fold_{fold}" for fold in range(len(scores))])
        return scores_df

    @staticmethod
    def shape_cv_preds(preds, labels, use_index=True):
        """Function to format the predictions from run_train_cv functions.
        Summarize the predictions at each fold as a pandas.DataFrame.

        Args:
            preds ([list]): [description]
            labels ([list, pandas.core.frame.DataFrame]): List of ground truth or y (target value dataframe with index of sample ID)
            use_index (bool, optional): Whether to use index of preds and labels. Defaults to True.

        Returns:
            preds_df [pandas.core.frame.DataFrame]: Summarized dataframe of the predictions at each fold.
        """        
        if use_index:
            preds_df = pd.merge(
                pd.concat(preds), pd.concat(labels), 
                how='left', left_index=True, right_index=True, 
                suffixes=('_preds', '_true')
            ).assign(
                fold=np.concatenate(
                    [np.repeat(fold, pred.shape[0]) 
                        for fold, pred in enumerate(preds)]
                )
            ) 
        else:
            preds_df = pd.merge(
                pd.concat(preds).reset_index(drop=True), pd.concat(labels).reset_index(drop=True), 
                how='left', left_index=True, right_index=True, 
                suffixes=('_preds', '_true')
            ).assign(
                fold=np.concatenate(
                    [np.repeat(fold, pred.shape[0]) 
                        for fold, pred in enumerate(preds)]
                )
            ) 
        return preds_df

    @staticmethod
    def shape_cv_logs(logs):
        """Function to format the logs from run_train_cv functions.
        Summarize the predictions at each fold as a pandas.DataFrame.

        Args:
            logs ([list]): List of the log dataframes at each fold.

        Returns:
            logs_df [pandas.core.frame.DataFrame]: Summarized dataframe of the predictions at each fold.
        """        
        logs_df = pd.concat([log.assign(fold=i) for i, log in enumerate(logs)])
        return logs_df
    
def get_file_type(path):
    pickle_extentions = set(['.pkl', '.pickle'])
    csv_extentions = set(['.csv'])
    ext = os.path.splitext(path)[1]
    if ext in pickle_extentions:
        return 'pkl'
    elif ext in csv_extentions:
        return 'csv'
    else:
        ValueError("Invalid file extention. Supported extentions are ['pkl', 'csv']")
    

def load_feature(feature_file, feature_dimension=26048, # 33280, 
                 blocksize='default', sample=370000, scheduler='processes'):
    """Support function to load feature dataframe from csv files.
    DASK is used to parallelize the reading of csv files.
    Like glob function, Wildcard (*) can be used in feature_file argument for example "feature_*.csv".
    CAUTION: The index of result dataframe will be processed by os.path.basename and file extension will be removed. The indices are then sorted alphabetically.

    Args:
        feature_file ([str]): File path to load.
        feature_dimension (int, optional): Number of the dimension of feature. Defaults to 33280.
        sample (int, optional): sample argument of ddf.read_csv function. Defaults to 370000.
            Guideline values of sample: 
                feature_dimension: 4608 -> sample: 256000
                feature_dimension: 33280 -> sample: 370000

    Returns:
        features [pandas.core.frame.DataFrame]: Loaded feature dataframe.
    """    
    # previous feature_dimension: 4608 -> sample: 256000 (default)
    # feature_dimension=33280 -> sample: 370000
    # read feature files and set index ------------------------------------
    file_type = get_file_type(feature_file)
    
    if file_type == 'csv':
        with ProgressBar():
            print(f"Reading feature files ({feature_file})...")
            dtypes = dict(zip(
                range(feature_dimension + 1), 
                ['str'] + ['float64'] * feature_dimension
            ))
            features = ddf.read_csv(feature_file, dtype=dtypes, blocksize=blocksize, sample=sample)
            features = features.compute(scheduler=scheduler)
            features.set_index(features.columns[0], inplace=True)
    elif file_type == 'pkl':
        features = pd.read_pickle(feature_file)  
    else:
        ValueError('Invalid file type')
    
#     features.index = [
#         os.path.basename(path.replace('.pdb', '')) for path in features.index
#     ]
    features.sort_index(inplace=True)
    
    return features

def load_label(label_file, label_colnames=['pKa_energy']): # label_colnames=['pKa', 'pKa_g1', 'pKa_g2']
    """Support function to load ground truth value dataframe from csv files.
    CAUTION: The indices of result dataframe are sorted alphabetically.

    Args:
        label_file ([type]): File path to load.
        label_colnames (list, optional): List of olumn names to load. Defaults to ['pKa_energy'].

    Returns:
        labels [pandas.core.frame.DataFrame]: Loaded ground truth valu dataframe.
    """    
    # read label file and set index ------------------------------------
    labels = pd.read_csv(label_file, index_col=0)
    labels = labels.loc[:, label_colnames]
    labels.sort_index(inplace=True)
    return labels