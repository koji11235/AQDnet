from abc import ABCMeta, abstractmethod
import os
import numpy as np
import pandas as pd
import dask.dataframe as ddf
from dask.diagnostics import ProgressBar
import joblib
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import concatenate
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.data import TFRecordDataset

import tensorflow as tf

from tensorflow_addons.layers import SpectralNormalization

import logging
import multiprocessing
import itertools
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
import palettable

import runner

##################################################################################
###################            Meta classes                    ###################
##################################################################################
class MyAbstractModel(metaclass=ABCMeta):
    """meta class of models below. Created with reference to Tensorflow model.
    Wrapper to use different classes of models (tensorflow, pytorch, lightgbm etc) in the same way in runner.Runner class functions.

    Args:
        metaclass ([type], optional): [description]. Defaults to ABCMeta.
    """    
    def __init__(self):
        return None 

    @abstractmethod
    def fit(self, X_train, y_train, X_valid=None, y_valid=None):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def evaluate(self, X, y):
        pass

    @abstractmethod
    def save_model(self):
        pass

    @abstractmethod
    def load_model(self):
        pass


##################################################################################
###################              Model classes                 ###################
##################################################################################


class ModelByTensorflow(MyAbstractModel):
    """Neural Network class of tensorflow. Wrapper of tensorflow model.fit, model.predict and model.evaluate.

    Args:
        MyAbstractModel ([type]): [description]
    """    
    def __init__(self, network_cls, **model_params):
        """[summary]

        Args:
            network_cls ([abc.ABCMeta]): NetworkStructure classs inherited from MyAbstractNetworkStructure
            model_params ([dict]): Acceptable model_params depends on network_cls.
        Returns:
            model [model.ModelByTensorflow]: model instance 

        Examples:
            >>> model = ModelByTensorflow(ElementwiseDNN4Tuning, **model_cls_params)
            >>> model.fit(X=X, y=y, validation_data=(X_valid, y_valid), **fit_params)
            >>> train_score, train_preds = model.evaluate(X=X, y=y)
            >>> valid_score, valid_preds = model.evaluate(X=X_valid, y=y_valid)
        """        
        self.model = None
        self.network_cls = network_cls
        self.model_params = model_params 
        self.network_structure = network_cls(**model_params)
        self.scaler = None
        self.history = None
        self.is_model_loaded = False
        return None 

    def _fit_by_pandas(self, X_train, y_train=None, X_valid=None, y_valid=None, callbacks=[], 
                        epochs=2, earlystop=20, batch_size_per_replica=128, 
                        use_mirrored_strategy=False, cuda_visible_device='-1', gpu_devices=None):
        """ fit function that accepts pandas.DataFrame as arguments.

        Args:
            X_train ([pandas.core.frame.DataFrame]): Input data. 
            y_train ([pandas.core.frame.DataFrame], optional): Target data. Defaults to None.
            X_valid ([pandas.core.frame.DataFrame], optional): Input data for validation. Defaults to None.
            y_valid ([pandas.core.frame.DataFrame], optional): Target data for validation. Defaults to None.
            callbacks (list, optional): List of callback functions to be given to model.fit(). Defaults to []. 
            use_mirrored_strategy (bool, optional): Whether to use multiple GPUs in training. Defaults to False. 
                ref: https://www.tensorflow.org/api_docs/python/tf/distribute/MirroredStrategy

        Returns:
            model [tensorflow.python.keras.engine.training.Model]: trained model of tensorflow.
            history [pandas.core.frame.DataFrame]: history of training.
        """  

        # preprocessing -------------------------------------------------------
        # pass X_train, X_valid as pandas dataframe
        # self.network_structure = self.network_cls(**self.model_params)
        X_train, y_train = self.network_structure.preprocess(X=X_train, y=y_train)
        X_valid, y_valid = self.network_structure.preprocess(X=X_valid, y=y_valid)

        # make callbacks -------------------------------------------------------
        # callbacks = self.create_callbacks(**params)
        early_stopping = EarlyStopping(
            monitor='val_loss', 
            min_delta=0.001, 
            patience=earlystop, 
            verbose=1,
            mode='auto',
            restore_best_weights=True ###CHANGED###
        )

        callbacks = callbacks + [early_stopping]
        
        # Create model -------------------------------------------------------
        model, strategy = self._create_model_on_or_not_on_mirrored_strategy(
            use_mirrored_strategy, gpu_devices
        )
        # print(f"use_mirrored_strategy: {use_mirrored_strategy}")
        # if use_mirrored_strategy:
        #     strategy = tf.distribute.MirroredStrategy(devices=gpu_devices)
        #     print(f"strategy.num_replicas_in_sync: {strategy.num_replicas_in_sync}")
        #     if self.model is None and not self.is_model_loaded:
        #         with strategy.scope():
        #             model = self.network_structure.create()
        #     elif self.is_model_loaded:
        #         with strategy.scope():
        #             model = self.network_structure.compile(self.model)
        # else:
        #     if self.model is None and not self.is_model_loaded:
        #         model = self.network_structure.create()
        #     elif self.is_model_loaded:
        #         model = self.network_structure.compile(self.model)

        # Train model -------------------------------------------------------
        if use_mirrored_strategy:
            global_batch_size = (batch_size_per_replica * strategy.num_replicas_in_sync)
        else:
            global_batch_size = batch_size_per_replica

        history = model.fit(
            x=X_train, y=y_train, 
            validation_data=(X_valid, y_valid),
            batch_size=global_batch_size, 
            epochs=epochs, 
            verbose=1, 
            callbacks=callbacks
        )
        return model, history

    def _fit_by_tfrecord(self, tfrecord_train, tfrecord_valid=None, callbacks=[], 
                        epochs=2, earlystop=20, batch_size_per_replica=128, 
                        use_mirrored_strategy=False, cuda_visible_device='-1', gpu_devices=None):
        """ fit function that accepts TFRecordDatasetV2 as arguments.

        Args:
            tfrecord_train ([tensorflow.python.data.ops.readers.TFRecordDatasetV2]): Input and Target data.
            tfrecord_valid ([tensorflow.python.data.ops.readers.TFRecordDatasetV2], optional): Input and Target data for validation. Defaults to None.
            callbacks (list, optional): List of callback functions to be given to model.fit(). Defaults to []. 
            use_mirrored_strategy (bool, optional): Whether to use multiple GPUs in training. Defaults to True. 
                ref: https://www.tensorflow.org/api_docs/python/tf/distribute/MirroredStrategy

        Returns:
            model [tensorflow.python.keras.engine.training.Model]: trained model of tensorflow.
            history [pandas.core.frame.DataFrame]: history of training.
        """    

        

        # make callbacks -------------------------------------------------------
        # callbacks = self.create_callbacks(**params)
        early_stopping = EarlyStopping(
            monitor='val_loss', 
            min_delta=0.001, 
            patience=earlystop, 
            verbose=1,
            mode='auto',
            restore_best_weights=True ###CHANGED###
        )
        
        callbacks = callbacks + [early_stopping]
        
        # Create model -------------------------------------------------------
        model, strategy = self._create_model_on_or_not_on_mirrored_strategy(
            use_mirrored_strategy, gpu_devices
        )
        
        # set batch size -------------------------------------------------------
        if use_mirrored_strategy:
            global_batch_size = (batch_size_per_replica * strategy.num_replicas_in_sync)
        else:
            global_batch_size = batch_size_per_replica

        # preprocessing -------------------------------------------------------
        ### (batching -> preprocessing) is three times faster than (preprocessing -> batching)
        dataset_train = tfrecord_train.batch(global_batch_size)
        if tfrecord_valid is not None:
            dataset_valid = tfrecord_valid.batch(global_batch_size)

        dataset_train = self.network_structure.preprocess(dataset_train)
        if tfrecord_valid is not None:
            dataset_valid = self.network_structure.preprocess(dataset_valid)
        else:
            dataset_valid = None

        # set prefetch size -------------------------------------------------------
        dataset_train = dataset_train.prefetch(tf.data.experimental.AUTOTUNE) # 1
        if dataset_valid is not None:
            dataset_valid = dataset_valid.prefetch(tf.data.experimental.AUTOTUNE) # 1
        
        print("global_batch_size: ", global_batch_size)

        # Train model -------------------------------------------------------
        history = model.fit(
            dataset_train, 
            validation_data=dataset_valid,
            batch_size=global_batch_size, 
            epochs=epochs, 
            verbose=1, 
            callbacks=callbacks
        )
        return model, history

    def fit(self, X, y=None, validation_data=None, callbacks=[], 
            epochs=2, earlystop=20, batch_size_per_replica=128, use_mirrored_strategy=False, cuda_visible_device='-1'):
        """ fit function that run _fit_by_pandas or _fit_by_tfrecord depending on the data type

        Args:
            X : Input data. Pandas DataFrame or TFRecordDatasetV2 can be accepted.
            y ([pandas.DataFram], optional): Target data. If the type of X is TFRecordDatasetV2, please set this to None. Defaults to None.
            validation_data (optional): Input and Target data for validation. 
                tuple of pandas.DataFrams (example: tuple(X_valid, y_valid)) or TFRecordDatasetV2 can be accepted. Defaults to None.
            callbacks (list, optional): List of callback functions to be given to model.fit(). Defaults to []. 
            use_mirrored_strategy (bool, optional): Whether to use multiple GPUs in training. Defaults to True. 
                ref: https://www.tensorflow.org/api_docs/python/tf/distribute/MirroredStrategy

        Raises:
            TypeError: Raised if the types of X, y and validation_data is not  pandas DataFrame or TFRecordDatasetV2.

        Returns:
            self [type]: self with trained model in self.model and training history in self.log
        """        
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_device
        gpu_devices = ['/gpu:' + device for device in cuda_visible_device.split(',') if not device == '-1']
        print(f"cuda_visible_device: {cuda_visible_device}")
        print(f"os.environ['CUDA_VISIBLE_DEVICES']: {os.environ['CUDA_VISIBLE_DEVICES']}")
        print(f"gpu_devices: {gpu_devices}, NUM_GPUS: {len(gpu_devices)}")

        fit_params = dict(
            epochs=epochs,
            earlystop=earlystop,
            batch_size_per_replica=batch_size_per_replica,
            use_mirrored_strategy=use_mirrored_strategy, 
            cuda_visible_device=cuda_visible_device, 
            gpu_devices=gpu_devices
        )

        if isinstance(X, TFRecordDataset) and y is None:
            model, history = self._fit_by_tfrecord(X, validation_data, callbacks, **fit_params)
        elif isinstance(X, pd.core.frame.DataFrame) and isinstance(y, pd.core.frame.DataFrame):
            X_valid, y_valid = validation_data
            model, history = self._fit_by_pandas(X, y, X_valid, y_valid, callbacks, **fit_params)
        else:
            raise TypeError('''Feature(X_train, X_valid) and label(y_train, y_valid) must be pandas.DataFrame or TFRecordDataset.
            Please check the type of your data.
            CAUTION: If X is TFRecordDataset, y must be None.''')
            
        # Save model and log -------------------------------------------------------
        self.log = self._make_log_df(history)
        self.model = model
        return self

    def _create_model_on_or_not_on_mirrored_strategy(self, use_mirrored_strategy=False, gpu_devices=None):

        print(f"use_mirrored_strategy: {use_mirrored_strategy}")
        
        if use_mirrored_strategy:
            strategy = tf.distribute.MirroredStrategy(devices=gpu_devices)
            print(f"strategy.num_replicas_in_sync: {strategy.num_replicas_in_sync}")
            if self.model is None and not self.is_model_loaded:
                with strategy.scope():
                    model = self.network_structure.create()
            elif self.is_model_loaded:
                with strategy.scope():
                    model = self.network_structure.compile(self.model)
        else:
            strategy = None
            if self.model is None and not self.is_model_loaded:
                model = self.network_structure.create()
            elif self.is_model_loaded:
                model = self.network_structure.compile(self.model)
        return model, strategy

    def _make_log_df(self, history):
        """function to shape log dataframe.

        Args:
            history ([type]): history returned by self._fit_by_tfrecord or self._fit_by_pandas.

        Returns:
            log_df [pandas.core.frame.DataFrame]: shaped training log dataframe.
        """        
        log_df = pd.DataFrame(history.history)
        log_df["epochs"] = log_df.index
        log_df.set_index('epochs')
        return log_df

    def _predict_pandas(self, X):
        """Predict function that accepts pandas DataFrame as an argument

        Args:
            X ([pandas.core.frame.DataFrame or tuple of pandas.DataFrame]): Input data.

        Returns:
            preds [pandas.core.frame.DataFrame]: Predicted values.
        """        
        # Preprocess -------------------------------------------------------
        # index = X.index
        X_processed, _ = self.network_structure.preprocess(X=X, y=None)
        
        # Retrieve index -------------------------------------------------------
        if isinstance(X, tuple):
            index = X[0].index
        else:
            index = X.index

        # Predict -------------------------------------------------------
        preds = self.model.predict(X_processed)
        preds = self.network_structure.shape_preds(preds, index=index)
        # preds.index = self.network_structure.generate_index(X, batch_size=batch_size)

        return  preds    

    def _predict_tfrecord(self, X, batch_size=128):
        """ Predict function that accepts TFRecordDatasetV2 as an argument

        Args:
            X ([TFRecordDatasetV2]): Input data.

        Returns:
            preds [pandas.core.frame.DataFrame]: Predicted values.
        """        
        # Preprocess -------------------------------------------------------
        X_batched = X.batch(batch_size)

        X_processed = self.network_structure.preprocess(X_batched).prefetch(tf.data.experimental.AUTOTUNE)

        # Predict -------------------------------------------------------
        preds = self.model.predict(X_processed)

        preds = self.network_structure.shape_preds(preds, index=None)
        preds.index = self.network_structure.generate_index(X, batch_size=batch_size)

        return  preds   

    def predict(self, X, batch_size=128):
        """ Predict method that run _fit_by_pandas or _fit_by_tfrecord depending on the data type

        Args:
            X : Input data. Pandas DataFrame of TFRecordDatasetV2 can be accepted.

        Raises:
            TypeError: Raised if the types of X is not pandas DataFrame or TFRecordDatasetV2.

        Returns:
            preds [pandas.core.frame.DataFrame]: Predicted values.
        """        
        if isinstance(X, TFRecordDataset):
            return self._predict_tfrecord(X, batch_size)
        elif isinstance(X, pd.core.frame.DataFrame) or isinstance(X, tuple):
            return self._predict_pandas(X)
        else:
            raise TypeError('''Feature(X) must be pandas.DataFrame or TFRecordDataset.
            Please check the type of your data.''')

    def _evaluate_pandas(self, X, y):
        """Evaluation method that accepts pandas.DataFrame as an argument

        Args:
            X ([pandas.core.frame.DataFrame]): Input data.
            y ([pandas.core.frame.DataFrame]): Target data.

        Returns:
            score [dict]: Dict with metrics_name as key and metrics_value as value. 
                Returned dict of self.model.evaluate(return_dict=True).
                Returned metricses depend on the network_cls and its loss and metrics.
            preds [pandas.core.frame.DataFrame]: Predicted values.
        """        
        # TODO: Change the code 
        # to calculate Preds first, and then 
        # calculate the score based on the Preds.
        
        # use tensorflow.model.evaluate()
        # calculate scores -------------------------------------------------------
        X_processed, y_processed = self.network_structure.preprocess(X, y)
        score = self.model.evaluate(X_processed, y_processed, return_dict=True)

        # calculate preds -------------------------------------------------------
        preds = self._predict_pandas(X)

        return score, preds
        # preds = self.predict(X)
        # score = self.network_cls.calculate_metrics(y_true=y, y_preds=preds)

    def _evaluate_tfrecord(self, X, batch_size=128):
        """ Evaluation method that accepts TFRecordDatasetV2 as an argument

        Args:
            X ([TFRecordDatasetV2]): Input and Target data.

        Returns:
            score [dict]: Dict with metrics_name as key and metrics_value as value. 
                Returned dict of self.model.evaluate(return_dict=True).
                Returned metricses depend on the network_cls and its loss and metrics.
            preds [pandas.core.frame.DataFrame]: Predicted values.
        """        
        # TODO: Change the code 
        # to calculate Preds first, and then 
        # calculate the score based on the Preds.

        # Preprocess -------------------------------------------------------
        X_batched = X.batch(batch_size)
        X_processed = self.network_structure.preprocess(X_batched).prefetch(tf.data.experimental.AUTOTUNE)

        # calculate scores  -------------------------------------------------------
        score = self.model.evaluate(X_processed, return_dict=True)

        # calculate preds -------------------------------------------------------
        preds = self._predict_tfrecord(X, batch_size)
        # preds = self.network_structure.shape_preds(preds, index=None)

        # generate index and set preds index -------------------------------------------------------
        preds.index = self.network_structure.generate_index(X, batch_size=batch_size)

        return score, preds
        
    def evaluate(self, X, y=None, batch_size=128):
        
        """Evaluate method that run _fit_by_pandas or _fit_by_tfrecord depending on the data type

        Args:
            X : Input data. Pandas DataFrame of TFRecordDatasetV2 can be accepted.
            y ([Pandas DataFram], optional): Target data. If the type of X is TFRecordDatasetV2, please set this to None. Defaults to None.

        Raises:
            TypeError: Raised if the types of X is not pandas DataFrame or TFRecordDatasetV2.

        Returns:
            preds [pandas.core.frame.DataFrame]: Predicted values.
        """        
        if isinstance(X, TFRecordDataset) and y is None:
            score, preds = self._evaluate_tfrecord(X, batch_size=batch_size)
        elif isinstance(X, pd.core.frame.DataFrame) and isinstance(y, pd.core.frame.DataFrame):
            score, preds = self._evaluate_pandas(X, y)
            
        elif isinstance(X, tuple) and isinstance(y, pd.core.frame.DataFrame):
            score, preds = self._evaluate_pandas(X, y)
        else:
            raise TypeError('''Feature(X) and label(y) must be pandas.DataFrame or TFRecordDataset.
            Please check the type of your data.
            CAUTION: If X is TFRecordDataset, y must be None.''')
        return score, preds


    def save_model(self, model_path):
        """method to write the model object to file.

        Args:
            model_path ([str]): The path of the model to be saved.

        Returns:
            self [type]: 
        """        

        if not os.path.exists(os.path.dirname(model_path)):
            os.makedirs(os.path.dirname(model_path), exist_ok=True)

        # scaler related operations were moved to Runner class
        # if not os.path.exists(os.path.dirname(scaler_path)):
        #     os.makedirs(os.path.dirname(scaler_path), exist_ok=True)

        self.model.save(model_path)
        # scaler related operations were moved to Runner class
        # joblib.dump(self.scaler, scaler_path)
        return self

    def load_model(self, model_path):
        """method to load the model object from file.

        Args:
            model_path ([str]): The path of the model to be loaded.

        Returns:
            self [type]: 
        """        
        self.model = tf.keras.models.load_model(model_path, compile=False)
        self.model = self.network_structure.compile(self.model)
        # scaler related operations were moved to Runner class
        # self.scaler = joblib.load(scaler_path)
        self.is_model_loaded = True
        
        return self


