import os
import re
import glob
import numpy as np
import pandas as pd
import dask.dataframe as ddf
import tensorflow as tf
from dask.diagnostics import ProgressBar
import logging
import multiprocessing
from tqdm import tqdm
from lpcomp import LigandProteinComplex

import runner

class FeatureGenerator(object):
    """Generate AQDnet features from ligand-protein complex PDB files."""

    def __init__(self, lig_code="LGD",
                 distance_threshold_radial=4, distance_threshold_angular=4,
                 target_elements=["H", "C", "N", "O", "P", "S", "Cl", "DU"],
                 Rs_list_radial=[0.5, 1.17, 1.83, 2.5, 3.17, 3.83, 4.5, 5.17],
                 Rs_list_angular=[0.5, 1.17, 1.83, 2.5, 3.17, 3.83, 4.5, 5.17],
                 theta_list=[0, 0.785, 1.57, 2.355, 3.14, 3.925, 4.71, 5.495]):
        """Initialize feature-generation hyperparameters."""
        self.pdb_files = []
        self.lig_code = lig_code
        self.target_elements = target_elements
        # self.distance_threshold = distance_threshold
        self.distance_threshold_radial = distance_threshold_radial
        self.distance_threshold_angular = distance_threshold_angular
        self.Rs_list_radial = Rs_list_radial
        self.Rs_list_angular = Rs_list_angular
        self.theta_list = theta_list
        return None

    def get_params(self):
        """Return the current generator parameters as a dictionary."""
        # return dict(distance_threshold=self.distance_threshold,
        #         target_elements=self.target_elements,
        #         Rs_list_radial=self.Rs_list_radial,
        #         Rs_list_angular=self.Rs_list_angular,
        #         theta_list=self.theta_list)
        return self.__dict__

    def _generate_feature_name(self, pdb_file, mode='complex', source='onionnet'):
        """Generate feature-column names for a representative input structure."""
        cplx = LigandProteinComplex(
            pdb_file, 
            lig_code=self.lig_code,
            target_elements=self.target_elements,
            distance_threshold_radial=self.distance_threshold_radial,
            distance_threshold_angular=self.distance_threshold_angular,
            Rs_list_radial=self.Rs_list_radial,
            Rs_list_angular=self.Rs_list_angular,
            theta_list=self.theta_list
        )
        cplx.parsePDB(source=source)
        if mode == 'complex':
            radial_feature_name, _ = cplx.generate_radial_feature()
            angular_feature_name, _ = cplx.generate_angular_feature()
        elif mode == 'ligand':
            radial_feature_name, _ = cplx.generate_radial_feature_onlyligand()
            angular_feature_name, _ = cplx.generate_angular_feature_onlyligand()
        return radial_feature_name, angular_feature_name

    def _generate_feature(self, pdb_file, source='onionnet'):
        """Generate radial and angular complex features for one PDB file."""

        cplx = LigandProteinComplex(
            pdb_file, 
            lig_code=self.lig_code,
            target_elements=self.target_elements,
            distance_threshold_radial=self.distance_threshold_radial,
            distance_threshold_angular=self.distance_threshold_angular,
            Rs_list_radial=self.Rs_list_radial,
            Rs_list_angular=self.Rs_list_angular,
            theta_list=self.theta_list
        )
        cplx.parsePDB(source=source)

        _, radial_feature_value = cplx.generate_radial_feature()
        _, angular_feature_value = cplx.generate_angular_feature()
        return radial_feature_value, angular_feature_value

    def _wrapped_generate_feature(self, kwargs):
        """Wrapper that logs per-file feature generation progress."""
        pdb_file = kwargs['pdb_file']
        logging.debug(f"{pdb_file}: started")
        radial_feature_value, angular_feature_value = self._generate_feature(
            **kwargs)
        logging.debug(f"{pdb_file}: finished")
        return radial_feature_value, angular_feature_value

    def _wrapped_generate_feature_for_tqdm(self, kwargs):
        """Wrapper used with multiprocessing and `tqdm` progress bars."""
        radial_feature_value, angular_feature_value = self._generate_feature(
            **kwargs)
        return radial_feature_value, angular_feature_value

    def _generate_ligand_feature(self, pdb_file, source='onionnet'):
        """Generate radial and angular ligand-only features for one PDB file."""
        cplx = LigandProteinComplex(
            pdb_file, 
            lig_code=self.lig_code,
            target_elements=self.target_elements,
            distance_threshold_radial=self.distance_threshold_radial,
            distance_threshold_angular=self.distance_threshold_angular,
            Rs_list_radial=self.Rs_list_radial,
            Rs_list_angular=self.Rs_list_angular,
            theta_list=self.theta_list
        )
        cplx.parsePDB(source=source)

        _, radial_feature_value = cplx.generate_radial_feature_onlyligand()
        _, angular_feature_value = cplx.generate_angular_feature_onlyligand()
        return radial_feature_value, angular_feature_value

    def _wrapped_generate_ligand_feature_for_tqdm(self, kwargs):
        """Wrapper for ligand-only feature generation with `tqdm`."""
        radial_feature_value, angular_feature_value = self._generate_ligand_feature(
            **kwargs)
        return radial_feature_value, angular_feature_value

    def generate(self, input_file, mode='complex', num_cpu=1, source='onionnet'):
        """Generate features from PDB paths listed in `input_file`.

        Args:
            input_file: Text file containing one PDB path per line.
            mode: Feature type to generate. Must be `'complex'` or `'ligand'`.
            num_cpu: Number of worker processes used for feature generation.
            source: Parser mode forwarded to `LigandProteinComplex.parsePDB`.

        Returns:
            MyDataFrame: Feature matrix indexed by input PDB path.
        """

        # set logging config during multiprocessing ----------------------------------------------------------
        logging.basicConfig(level=logging.DEBUG,
                            format='%(processName)s: %(message)s')

        # load pdb file list from input file ----------------------------------------------------------
        self.pdb_files = pd.read_table(input_file, header=None)[0].tolist()

        # set hyper parameters of generate_feature ----------------------------------------------------------
        generate_feature_args = [
            {"pdb_file": pdb_file, 'source': source}
            for pdb_file in self.pdb_files
        ]
        self.radial_feature_name, self.angular_feature_name = \
            self._generate_feature_name(self.pdb_files[0], mode=mode, source=source)

        # parallel feature generation ----------------------------------------------------------
        with multiprocessing.Pool(num_cpu) as p:
            # features = p.map(self._wrapped_generate_feature,
            #                  generate_feature_args)
            if mode == 'complex':
                features = list(tqdm(
                    p.imap(
                        self._wrapped_generate_feature_for_tqdm,
                        generate_feature_args),
                    total=len(generate_feature_args)))
            elif mode == 'ligand':
                features = list(tqdm(
                    p.imap(
                        self._wrapped_generate_ligand_feature_for_tqdm,
                        generate_feature_args),
                    total=len(generate_feature_args)))
            else:
                raise ValueError("mode argument must be either 'complex' or 'ligand'.")
            logging.debug("Feature generation completed ...... ")

        return MyDataFrame(
            data=np.vstack([np.block([radial, angular])
                            for radial, angular in features]),
            index=self.pdb_files,
            columns=(self.radial_feature_name + self.angular_feature_name)
        )


class MyDataFrame(pd.DataFrame):
    """Thin `pandas.DataFrame` subclass with AQDnet export helpers."""

    def __init__(self, data=None, index=None, columns=None, dtype=None, copy=False):
        super().__init__(data=data, index=index, columns=columns, dtype=dtype, copy=copy)

    def to_csv_parallelized(self, output_file, single_file=False, npartitions=10, scheduler='processes'):
        """Write the dataframe as CSV with Dask-backed parallelization."""
        with ProgressBar():
            print_sentence = "Saving the dataset in a csv file" if single_file else "Saving the dataset in csv files"
            print(print_sentence)
            ddf.to_csv(
                df=ddf.from_pandas(self, npartitions=npartitions),
                filename=output_file,
                # scheduler='processes',
                compute_kwargs={'scheduler': scheduler},
                single_file=single_file
            )  # compute_kwargs={"scheduler": 'processes'}
        print(f"{output_file} was saved...")
        return None

    @staticmethod
    def feature2example(complex_feature, label, sample_id):
        """Convert one sample into a serialized TF Example schema."""
        return tf.train.Example(features=tf.train.Features(feature={
            'feature': _float_feature(complex_feature),
            'label': _float_feature(label),
            'sample_id': _bytes_feature(sample_id)
        }))

    def __to_tfrecords(self, output_file, label, label_colnames=['pKa']):
        """Write features and labels to a TFRecord file.

        Notes:
            This method is marked as deprecated in the original implementation.
        """
        # depreciated: tfrecords files generated by this function cause unknown error below.
        # InvalidArgumentError: Feature: complex_feature (data type: float) is required but could not be found. [[{{node ParseSingleExample/ParseExample/ParseExampleV2}}]]

        # feature = self.copy()
        feature_ = pd.DataFrame(self.values, index=self.index, columns=self.columns)
        print(feature_)
        print(feature_.dtypes, label.dtypes)
        feature = feature_.astype('float32')
        label = label.astype('float32')
        print(feature.dtypes, label.dtypes)
        common_idx = feature.index.intersection(label.index)
        feature = feature.loc[common_idx]

        label = label.loc[common_idx, label_colnames]
        feature.sort_index(inplace=True)
        label.sort_index(inplace=True)

        if not all(feature.index == label.index): 
            raise ValueError('feature and label have different indexes.')
        if len(feature.index) == 0 :
            raise ValueError('There are no features to write. Length of feature is 0.')
        
        sample_id = np.vectorize(lambda x: x.encode())(label.index.values)
        print(feature.shape, label.shape, len(sample_id))

        # write data to tfr_filename
        n_sample = sample_id.shape[0]
        with tf.io.TFRecordWriter(output_file) as writer:
            for i in tqdm(range(n_sample)):
                ex = MyDataFrame.feature2example(
                    complex_feature=feature.values[i], 
                    label=label.values[i], 
                    sample_id=sample_id[i]
                )
                writer.write(ex.SerializeToString())

        print(f"{output_file} was saved...")
        return None


def to_csv_parallelized(df, output_file, single_file=False, npartitions=10):
    """Write a dataframe to CSV files with Dask parallelization."""
    with ProgressBar():
        print_sentence = "Saving the dataset in a csv file" if single_file else "Saving the dataset in csv files"
        print(print_sentence)
        ddf.to_csv(
            df=ddf.from_pandas(df, npartitions=npartitions),
            filename=output_file,
            # scheduler='processes',
            compute_kwargs={'scheduler': 'processes'},
            single_file=single_file
        )  # compute_kwargs={"scheduler": 'processes'}
    print(f"{output_file} was saved...")
    return None

def _float_feature(value):
    """Build a TF float feature from an iterable of numeric values."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _bytes_feature(value):
    """Build a TF bytes feature from a byte string or eager tensor."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def csv_to_tfrecords(feature_csv_files, label_file, output_file, label_colnames=['pKa'], sample=370000):
    """Convert feature CSV shards and labels into a single TFRecord file."""
    feature = runner.load_feature(feature_csv_files, sample=sample).astype('float32')
    label = runner.load_label(label_file, label_colnames=label_colnames).astype('float32')

    common_idx = feature.index.intersection(label.index)
    feature = feature.loc[common_idx]
    label = label.loc[common_idx, label_colnames]

    if not all(feature.index == label.index): 
            raise ValueError('feature and label have different indexes.')
    if len(feature.index) == 0 :
        raise ValueError('There are no features to write. Length of feature is 0.')

    # create byte type sample_id
    sample_id = np.vectorize(lambda x: x.encode())(label.index.values)

    # write data to tfr_filename
    n_sample = sample_id.shape[0]
    with tf.io.TFRecordWriter(output_file) as writer:
        for i in tqdm(range(n_sample)):
            ex = record2example(
                cmp_feature=feature.values[i], 
                label=label.values[i], 
                sample_id=sample_id[i]
            )
            writer.write(ex.SerializeToString())
    print(f"{output_file} was saved...")
    return None

def bothfeature2example(cmp_feature, lig_feature, label, sample_id):
    """Convert complex and ligand features for one sample into a TF Example."""
    return tf.train.Example(features=tf.train.Features(feature={
        'cmp_feature': _float_feature(cmp_feature),
        'lig_feature': _float_feature(lig_feature),
        'label': _float_feature(label),
        'sample_id': _bytes_feature(sample_id)
    }))

# def record2example(feature_dict):
#     return tf.train.Example(features=tf.train.Features(feature=feature_dict))

def record2example(cmp_feature, label, sample_id):
    """Convert one complex-feature sample into a TF Example."""
    return tf.train.Example(features=tf.train.Features(feature={
        'complex_feature': _float_feature(cmp_feature),
        'label': _float_feature(label),
        'sample_id': _bytes_feature(sample_id)
    }))

def generate_feature(input_file, output_files, mode='complex', num_cpu=1, source='onionnet'):
    """Generate features from an input list and save them as CSV."""
    fg = FeatureGenerator()
    dataset = fg.generate(input_file, mode=mode, num_cpu=num_cpu, source=source)
    dataset.to_csv_parallelized(output_files)
    return dataset

def get_basename_without_extention(index):
    """Return file basenames without extensions for each input path."""
    return [os.path.splitext(os.path.basename(path))[0] for path in index]

def get_pdbid(index):
    """Extract PDB IDs from AQDnet complex filenames."""
    return [os.path.splitext(os.path.basename(path))[0].replace('_complex', '') for path in index]

def generate_both_feature_to_tfrecords(input_file, label, output_tfrecords_file, 
                                        output_cmp_csv=None, output_lig_csv=None, 
                                        num_cpu=1, source='onionnet', index_process_fn=None):
    """Generate complex and ligand features, then save them to TFRecord.

    Optionally writes the intermediate complex and ligand feature CSV files.
    """
    if index_process_fn is None:
        index_process_fn = get_basename_without_extention

    fg = FeatureGenerator()

    # generate complex feature ------------------------------------------------------------
    cmp_dataset = fg.generate(input_file, mode='complex', num_cpu=num_cpu, source=source)
    cmp_dataset.index = index_process_fn(cmp_dataset.index)

    # generate ligand feature ------------------------------------------------------------
    lig_dataset = fg.generate(input_file, mode='ligand', num_cpu=num_cpu, source=source)
    lig_dataset.index = index_process_fn(lig_dataset.index)

    # check the index of cmp, lig and label ------------------------------------------------------------
    common_idx = cmp_dataset.index.intersection(label.index)
    cmp_dataset = cmp_dataset.loc[common_idx]
    lig_dataset = lig_dataset.loc[common_idx]
    label = label.loc[common_idx]
    cmp_dataset.sort_index(inplace=True)
    lig_dataset.sort_index(inplace=True)
    label.sort_index(inplace=True)
    if not all(cmp_dataset.index == label.index) or not all(lig_dataset.index == label.index): 
        raise ValueError('feature and label have different indexes.')
    if len(cmp_dataset.index) == 0 or len(lig_dataset.index) == 0 :
        raise ValueError('There are no features to write. Length of feature is 0.')

    # save to csv ------------------------------------------------------------
    if output_cmp_csv is not None:
        to_csv_parallelized(cmp_dataset, output_cmp_csv)

    if output_lig_csv is not None:
        to_csv_parallelized(lig_dataset, output_lig_csv)
        
    # save to tfrecord ------------------------------------------------------------
    sample_id = np.vectorize(lambda x: x.encode())(label.index.values)
    n_sample = sample_id.shape[0]
    with tf.io.TFRecordWriter(output_tfrecords_file) as writer:
        for i in tqdm(range(n_sample)):
            ex = bothfeature2example(
                cmp_feature=cmp_dataset.values[i], 
                lig_feature=lig_dataset.values[i], 
                label=label.values[i], 
                sample_id=sample_id[i]
            )
            writer.write(ex.SerializeToString())
    return None

def glob_re(strings, pattern):
    """Filter strings by a regular-expression pattern."""
    return list(filter(re.compile(pattern).match, strings))

def make_generate_inputfile(directory, filename, pattern="\w{4}_complex.pdb$"):
    """Create a text file listing PDB files that match `pattern`.

    Args:
        directory: Directory to search for input PDB files.
        filename: Output text file that will contain one path per line.
        pattern: Regular expression matched against candidate paths.

    Examples:
        >>> directory = '../1.Input/test_CASF2013_coreset/'
        >>> fg_inputfile_path = '../1.Input/test_fg_inputfile.dat'
        >>> make_generate_inputfile(directory, fg_inputfile_path, pattern="\w{4}_complex.pdb$")
        
    """    
    pattern = os.path.join(directory, pattern)
    path_list = glob_re(glob.glob(os.path.join(directory, "*")), pattern)
    with open(filename, mode='w') as f:
        f.write('\n'.join(path_list))
    return None 

def mother_params_to_fg_params(mother_params):
    """Convert a high-level parameter dict into `FeatureGenerator` kwargs."""

    distance_threshold_radial = mother_params['distance_threshold_radial']
    distance_threshold_angular = mother_params['distance_threshold_angular']
    Rs_radial_step = mother_params['Rs_radial_step']
    Rs_angular_step = mother_params['Rs_angular_step']
    n_theta = mother_params['n_theta']
    target_elements = mother_params.get('target_elements', ["H", "C", "N", "O", "P", "S", "Cl", "DU"])

    # distance_threshold_radial = mother_params.get('distance_threshold_radial', 4)
    # distance_threshold_angular = mother_params.get('distance_threshold_angular', 4)
    # Rs_radial_step = mother_params.get('Rs_radial_step', 0.5)
    # Rs_angular_step = mother_params.get('Rs_angular_step', 0.5)
    # n_theta = mother_params.get('n_theta', 8)

    Rs_list_radial = np.arange(0.5, distance_threshold_radial, Rs_radial_step).tolist()
    Rs_list_angular = np.arange(0.5, distance_threshold_angular, Rs_angular_step).tolist()
    theta_list = np.linspace(0, np.pi*2, n_theta, endpoint=False, dtype='Float32').tolist()

    fg_params = dict(
        distance_threshold_radial=distance_threshold_radial, 
        distance_threshold_angular=distance_threshold_angular,
        target_elements=target_elements,
        Rs_list_radial=Rs_list_radial,
        Rs_list_angular=Rs_list_angular,
        theta_list=theta_list
    )
    return fg_params
    
def get_default_param():
    """Return the default feature-generation parameter set."""
    default_param = dict(
        distance_threshold_radial=4, 
        distance_threshold_angular=4,
        target_elements=["H", "C", "N", "O", "P", "S", "Cl", "DU"],
        Rs_list_radial=[0.5, 1.17, 1.83, 2.5, 3.17, 3.83, 4.5, 5.17],
        Rs_list_angular=[0.5, 1.17, 1.83, 2.5, 3.17, 3.83, 4.5, 5.17],
        theta_list=[0, 0.785, 1.57, 2.355, 3.14, 3.925, 4.71, 5.495]
    )
    return default_param
# https://github.com/microsoft/LightGBM/blob/master/python-package/lightgbm/engine.py
