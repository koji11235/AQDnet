# import os
# from sklearn.externals import joblib


# class Util:

#     @classmethod
#     def dump(cls, value, path):
#         os.makedirs(os.path.dirname(path), exist_ok=True)
#         joblib.dump(value, path, compress=True)

#     @classmethod
#     def load(cls, path):
#         return joblib.load(path)


import pandas as pd
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import palettable

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

import os

import runner




def save_runner_result(runner, result, output_dir):
    model_and_log_path_filepath = os.path.join(output_dir, 'model_and_log_paths.csv')
    prediction_filepath = os.path.join(output_dir, 'prediction.csv')
    valid_indices_filepath = os.path.join(output_dir, 'valid_indices.pkl')
    scores_filepath = os.path.join(output_dir, 'scores.csv')
    runner_result_filepath = os.path.join(output_dir, 'runner_result.pkl')
    
    paths = pd.DataFrame(runner.paths).transpose()
    scores, preds, valid_indices, logs  = result

    paths.to_csv(model_and_log_path_filepath)
    preds.to_csv(prediction_filepath)
    scores.to_csv(scores_filepath)
    pd.to_pickle(valid_indices, valid_indices_filepath)
    pd.to_pickle((scores, preds, valid_indices, logs, paths), runner_result_filepath)
    return None

def make_groups_by_pdbid(y):
    groups = y.index.str.extract('smina_(.{4})_')[0].values
    le = LabelEncoder()
    le.fit(groups)
    groups_encoded = le.transform(groups)
    return groups

# def feature_float_list(l):
#     return tf.train.Feature(float_list=tf.train.FloatList(value=l))

# def record2example(r):
#     return tf.train.Example(features=tf.train.Features(feature={
#         "X": feature_float_list(r[0:-1]),
#         "y": feature_float_list([r[-1]])
#     }))

# def write_feature_to_tfrecords(feature_files, label_file, tfr_filename, label_colnames=['pKa_energy']):
#     # load features and labels
#     feature = runner.load_feature(feature_files).astype('float32')
#     label = runner.load_label(label_file, label_colnames=label_colnames).astype('float32')

#     # remove not common index
#     common_idx = feature.index.intersection(label.index)
#     feature = feature.loc[common_idx]
#     label = label.loc[common_idx]

#     # check whether feature and label have the same index.
#     if not all(feature.index == label.index):
#         raise ValueError('X and y have different indexes.')
    
#     # check whether len of feature is not 0
#     if len(feature.index) == 0:
#         raise ValueError('There are no features to write. Length of feature is 0.')

#     # concatenate feature and label
#     data = np.c_[feature.values, label.values]

#     # write data to tfr_filename
#     with tf.io.TFRecordWriter(tfr_filename) as writer:
#         for r in data:
#             ex = record2example(r)
#             writer.write(ex.SerializeToString())
#     return None


class ConvertFeatureCSV2TFRecords(object):
    def __init__(self):
        return None
        
    # TFRecord file generation related methods
    @staticmethod
    def _float_feature(value):
        """return float_list from float / double """
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    @staticmethod
    def _bytes_feature(value):
        """return byte_list from string / byte """
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    @staticmethod
    def record2example(cmp_feature, label, sample_id):
        return tf.train.Example(features=tf.train.Features(feature={
            'complex_feature': ConvertFeatureCSV2TFRecords._float_feature(cmp_feature),
            'label': ConvertFeatureCSV2TFRecords._float_feature(label),
            'sample_id': ConvertFeatureCSV2TFRecords._bytes_feature(sample_id)
        }))

    @staticmethod
    def convert_feature_csv_to_tfrecords(complex_feature_files, tfr_filename, 
                                         label_file=None, label_colnames=['pKa_energy'], 
                                    feature_dimension=26048, dask_sample=290000, 
                                    # prev ver feature_dimension=33280, dask_sample=370000,
                                    scheduler='processes', feature_index_str_replace=None): # feature_index_str_replace='_complex
        # load features and labels
        feature_cmp = runner.load_feature(
            complex_feature_files, 
            scheduler=scheduler,
            feature_dimension=feature_dimension,
            sample=dask_sample, 
        ).astype('float32')
        
        if label_file is not None:
            label = runner.load_label(label_file, label_colnames=label_colnames).astype('float32')
        else:
            label = pd.DataFrame(np.nan, index=feature_cmp.index, columns=['Dummy_Label'])

        # process feature_cmp index
        if feature_index_str_replace is not None:
            feature_cmp.index = [idx.replace(feature_index_str_replace, '') for idx in feature_cmp.index]

        # remove not common index
        common_idx = feature_cmp.index.intersection(label.index)
        feature_cmp = feature_cmp.loc[common_idx]
        label = label.loc[common_idx]

        # check feature 
        if not all(feature_cmp.index == label.index): 
            raise ValueError('feature_cmp and label have different indexes.')
        if len(feature_cmp.index) == 0:
            raise ValueError('There are no features to write. Length of feature is 0.')

        # create byte type sample_id
        sample_id = np.vectorize(lambda x: x.encode())(label.index.values)
        
        # write data to tfr_filename
        n_sample = sample_id.shape[0]
        with tf.io.TFRecordWriter(tfr_filename) as writer:
            for i in tqdm(range(n_sample)):
                ex = ConvertFeatureCSV2TFRecords.record2example(
                    cmp_feature=feature_cmp.values[i], 
                    label=label.values[i], 
                    sample_id=sample_id[i]
                )
                writer.write(ex.SerializeToString())
        return None

#     @staticmethod
#     def convert_feature_csv_to_tfrecords(complex_feature_files, label_file, tfr_filename, 
#                                          label_colnames=['pKa_energy'], feature_dimension=26048, 
#                                          dask_sample=290000, scheduler='processes'):
#         ConvertFeatureCSV2TFRecords._convert_feature_csv_to_tfrecords(
#             complex_feature_files, 
#             label_file, 
#             tfr_filename, 
#             label_colnames=label_colnames,
#             feature_dimension=feature_dimension,
#             dask_sample=dask_sample, 
#             scheduler=scheduler
#         )
#         print(f"{tfr_filename} was saved...")
#         return None


################ result analysis ################

def make_result_df(eval_results, metrics=['RMSE', 'PCC', 'PCC_RMSE']):
    result_df = pd.DataFrame()
    for condition, result in eval_results.items():
        scores = {k: v[0] for (k, v) in result.items()}
        eval_result_df = pd.DataFrame(scores.values(), index=scores.keys())
        eval_result_df = eval_result_df.loc[: ,metrics]
        eval_result_df.reset_index(inplace=True)
        eval_result_df.rename(columns={'index': 'dataset'}, inplace=True)
        eval_result_df = eval_result_df.assign(condition=condition)
        result_df = result_df.append(eval_result_df)
    return result_df

def plot_metrics(result_df, x='condition', metrics=['PCC', 'RMSE', 'PCC_RMSE'], 
                 save=True, output_path="image/High/{metric}.png", height=10, aspect=0.5):
    for metric in metrics:
        g = sns.catplot(data=result_df, kind='bar', x=x, y=metric, 
                        col='dataset', height=height, aspect=aspect)
        for ax in g.axes.flat:
            for label in ax.get_xticklabels():
                label.set_rotation(90)
        plt.show()
        if save:
            g.savefig(output_path.format(metric=metric))
    return None


def make_label_and_preds(eval_results, dataset_to_label, columns=['pKa_energy'], only_test=True):
    
#     dataset_to_label_path = {
#         'train': '../../1.Input/4.Label/cPDHK_2_resolution/High_resolution/train_set.csv', 
#         'valid': '../../1.Input/4.Label/cPDHK_2_resolution/High_resolution/valid_set.csv', 
#         'test': '../../1.Input/4.Label/cPDHK_2_resolution/High_resolution/test_set.csv', }
#     dataset_to_label = {dataset_name: pd.read_csv(path, index_col=0) 
#                         for dataset_name, path in dataset_to_label_path.items()}

    all_label_and_preds = pd.DataFrame()
    for condition, result in eval_results.items():
        for dataset, label in dataset_to_label.items():
            if only_test and dataset != 'test':
                continue
            preds = result[dataset][1]
            preds.rename(columns={column: column + '_preds' for column in columns}, inplace=True)
            label_and_preds = pd.merge(preds, label, how='left', left_index=True, right_index=True)
            label_and_preds = label_and_preds.assign(dataset=dataset, condition=condition)
            all_label_and_preds = all_label_and_preds.append(label_and_preds)
            # print(all_label_and_preds.head(2))

    all_label_and_preds['pdb_or_psilo'] = all_label_and_preds.index.str.extract('smina_(psiloPDHK)?(\w{4})_')[1].values
    return all_label_and_preds


def plot_scatter_label_vs_preds(all_label_and_preds, 
                                label='pKa_energy', preds='pKa_energy_preds', 
                                save=True, output=None, alpha=0.05):
    
    scatter_facet = sns.FacetGrid(all_label_and_preds, 
                                  col="dataset", row='condition', hue='condition') 
    scatter_facet.map(sns.scatterplot, label, preds, alpha=alpha)
    [plt.setp(ax.texts, text="") for ax in scatter_facet.axes.flat]
    scatter_facet.set_titles(col_template = '', row_template = '', template='{row_name}:{col_name}')
    plt.show()
    if save:
        scatter_facet.savefig(output)
    return scatter_facet