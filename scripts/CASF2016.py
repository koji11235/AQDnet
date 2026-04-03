import os
import glob
import re
import pandas as pd
import tensorflow as tf
from structure import ElementwiseDNN
from model import ModelByTensorflow

# coreset_path = '/home/koji-shiota/project/AQDNet/1.Input/5.Dataset/CASF2016/CoreSet/Ver1/tfrecords/feature_coreset.tfrecords'
# docking_path = '/home/koji-shiota/project/AQDNet/1.Input/5.Dataset/CASF2016/Docking/Ver1/tfrecords/feature_docking.tfrecords'
# screening_paths = glob.glob('/home/koji-shiota/project/AQDNet/1.Input/5.Dataset/CASF2016/Screening/Ver1/tfrecords/*.tfrecords')


def load_model(model_path, model_params):
    """Load a trained AQDnet model from disk."""
    model = ModelByTensorflow(network_cls=ElementwiseDNN, **model_params)
    model.load_model(model_path)
    return model

def shape_preds_coreset(preds_coreset, output_dir, score_column='pKa_energy', index_pattern='(\w{4})_complex'):
    """Write coreset predictions in CASF-2016 scoring format."""
    _preds_coreset = preds_coreset.copy(deep=True)
    _preds_coreset = _preds_coreset.rename(columns={score_column: 'score'})
    _preds_coreset['#code'] = _preds_coreset.index.str.extract(index_pattern)[0].values
    _preds_coreset = _preds_coreset.reindex(columns=['#code', 'score'])
    _preds_coreset.to_csv(os.path.join(output_dir, 'scoring.dat'), index=False, sep='\t')
    return None

def shape_preds_docking(preds_docking, preds_coreset, output_dir, output_dir_wo_native, score_column='pKa_energy', 
                        docking_index_pattern='(\w{4}_\d+)_complex', coreset_index_pattern='(\w{4})_complex'):
    """Write docking predictions with and without native poses in CASF-2016 format."""
    # without Native Ligand Pose
    _preds_docking = preds_docking.copy(deep=True)
    _preds_coreset = preds_coreset.copy(deep=True)
    
    _preds_docking = _preds_docking.rename(columns={score_column: 'score'})
    _preds_docking['#code'] = _preds_docking.index.str.extract(docking_index_pattern)[0].values 
    _preds_docking['pdbid'] = _preds_docking['#code'].str.split('_', expand=True)[0].values
    
    gb = _preds_docking.groupby('pdbid')
    for pdbid in gb.groups.keys():
        gb.get_group(pdbid).reindex(columns=['#code', 'score'])\
            .to_csv(os.path.join(output_dir_wo_native, f"{pdbid}_score.dat"), index=False, sep='\t')
        
    # with Native Ligand Pose
    _preds_coreset = _preds_coreset.rename(columns={score_column: 'score'})
    _preds_coreset['#code'] = _preds_coreset.index.str.extract(coreset_index_pattern)[0].values + '_ligand'
    _preds_coreset['pdbid'] = _preds_coreset['#code'].str.split('_', expand=True)[0].values

    _preds_docking = pd.concat([_preds_docking, _preds_coreset])
    
    gb = _preds_docking.groupby('pdbid')
    for pdbid in gb.groups.keys():
        gb.get_group(pdbid).reindex(columns=['#code', 'score'])\
            .to_csv(os.path.join(output_dir, f"{pdbid}_score.dat"), index=False, sep='\t')
    return None

def shape_preds_screening_not_separated(preds_screening, output_dir, protein_pdbid, 
                          score_column='pKa_energy', index_pattern='\w{4}_(\w{4}_ligand_\d+)_complex'):
    """Write screening predictions for a single TFRecord shard."""
    _preds_screening = preds_screening.copy(deep=True)
    _preds_screening = _preds_screening.rename(columns={score_column: 'score'})
    _preds_screening['#code_ligand_num'] = _preds_screening.index.str.extract(index_pattern)[0].values
    _preds_screening.reindex(columns=['#code_ligand_num', 'score'])\
        .to_csv(os.path.join(output_dir, f"{protein_pdbid}_score.dat"), index=False, sep='\t')
    return None

def shape_preds_screening(preds_screening, output_file, 
                          score_column='pKa_energy', index_pattern='\w{4}_(\w{4}_ligand_\d+)_complex'):
    """Write merged screening predictions in CASF-2016 format."""
    _preds_screening = preds_screening.copy(deep=True)
    _preds_screening = _preds_screening.rename(columns={score_column: 'score'})
    _preds_screening['#code_ligand_num'] = _preds_screening.index.str.extract(index_pattern)[0].values
    _preds_screening.reindex(columns=['#code_ligand_num', 'score'])\
        .to_csv(output_file, index=False, sep='\t')
    return None


def prepare_scoring(model, coreset_path, output_dir, batch_size=300, 
                    score_column='pKa_energy', index_pattern='(\w{4})_complex'):
    """Run inference on the CASF-2016 coreset and save `scoring.dat`."""
    X_coreset = tf.data.TFRecordDataset([coreset_path])
    preds_coreset = model.predict(X_coreset, batch_size=batch_size)
    shape_preds_coreset(preds_coreset, output_dir, score_column, index_pattern)
    return None

def prepare_docking(model, docking_path, coreset_path, output_dir, output_docking_dir_wo_native, batch_size=10000, score_column='pKa_energy', 
                        docking_index_pattern='(\w{4}_\d+)_complex', coreset_index_pattern='(\w{4})_complex'):
    """Run docking inference and save CASF-2016 docking input files."""
    X_docking = tf.data.TFRecordDataset([docking_path])
    preds_docking = model.predict(X_docking, batch_size=batch_size)
    
    X_coreset = tf.data.TFRecordDataset([coreset_path])
    preds_coreset = model.predict(X_coreset, batch_size=batch_size)
    
    shape_preds_docking(preds_docking, preds_coreset, output_dir, output_docking_dir_wo_native,
                        score_column, docking_index_pattern, coreset_index_pattern)
    return None

def prepare_screening(model, screening_paths, output_dir, batch_size=10000, score_column='pKa_energy',  
                      tfrecords_pattern="feature_screening_(\w{4})(_\d+)?.tfrecords", index_pattern='\w{4}_(\w{4}_ligand_\d+)_complex'):
    """Run screening inference and save per-protein CASF-2016 score files.

    This function supports both a single TFRecord per protein and multiple
    sharded TFRecords per protein.
    """
#     print(tfrecords_pattern)
#     print(os.path.basename(screening_paths[0]))
#     print(re.match(tfrecords_pattern, os.path.basename(screening_paths[0])))
    
    if re.match(tfrecords_pattern, os.path.basename(screening_paths[0]))[2] is None:
        for screening_path in screening_paths:
            print(f"{os.path.basename(screening_path)} started...")
            X_screening = tf.data.TFRecordDataset(screening_path)
            preds_screening = model.predict(X_screening, batch_size=10000)
            protein_pdbid = re.match(tfrecords_pattern, os.path.basename(screening_path))[1]
            shape_preds_screening_not_separated(preds_screening, output_dir, protein_pdbid, score_column, index_pattern)
            print(f"{os.path.basename(screening_path)} finished...")
            
    elif re.match(tfrecords_pattern, os.path.basename(screening_paths[0]))[2] is not None:
        screening_dir = os.path.dirname(screening_paths[0])
        protein_pdbids = sorted(list(set([re.match(tfrecords_pattern, 
                                                   os.path.basename(screening_path))[1] 
                                          for screening_path in screening_paths])))
        for protein_pdbid in protein_pdbids:
            tfrecords_path = os.path.join(screening_dir, f"feature_screening_{protein_pdbid}_*.tfrecords")
            print(f"{os.path.basename(tfrecords_path)} started...")
            tfrecords_files = glob.glob(tfrecords_path)
            output_file = os.path.join(output_dir, f"{protein_pdbid}_score.dat")
            X_screening = tf.data.TFRecordDataset(tfrecords_files)
            preds_screening = model.predict(X_screening, batch_size=batch_size)
            shape_preds_screening(preds_screening, output_file, score_column, index_pattern)
            print(f"{os.path.basename(tfrecords_path)} finished...")
        
    return None
    
def prepare(model_path, model_params, coreset_path, docking_path, screening_paths, output_dir, 
            batch_size=10000, score_column='pKa_energy', run_screening_power=True):
    """Prepare CASF-2016 evaluation files from AQDnet predictions.

    This function loads a trained model, runs inference on the CASF-2016
    coreset, docking, and optionally screening TFRecords, then writes the
    prediction files expected by the external CASF-2016 evaluation scripts.
    It does not compute the evaluation metrics itself.

    Args:
        model_path: Path to the trained AQDnet model checkpoint or directory.
        model_params: Keyword arguments used to initialize `ModelByTensorflow`.
        coreset_path: TFRecord file for the CASF-2016 coreset.
        docking_path: TFRecord file for the CASF-2016 docking set.
        screening_paths: TFRecord files for the CASF-2016 screening set.
        output_dir: Directory where formatted evaluation files are written.
        batch_size: Batch size used for model inference.
        score_column: Prediction column to export as the CASF score.
        run_screening_power: Whether to generate screening-power inputs.

    Outputs:
        - `scoring.dat` in `output_dir`
        - docking score files in `output_dir/Docking`
        - docking score files without native poses in
          `output_dir/Docking_wo_native`
        - screening score files in `output_dir/Screening` when enabled
    """
    output_docking_dir = os.path.join(output_dir, 'Docking')
    output_docking_dir_wo_native = os.path.join(output_dir, 'Docking_wo_native')
    output_screening_dir = os.path.join(output_dir, 'Screening')
    
    for directory in [output_docking_dir, output_docking_dir_wo_native, output_screening_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    model = load_model(model_path, model_params)
    prepare_scoring(model, coreset_path, output_dir, batch_size, 
                    score_column, index_pattern='(\w{4})_complex')
    prepare_docking(model, docking_path, coreset_path, output_docking_dir, 
                    output_docking_dir_wo_native, batch_size, score_column, 
                    docking_index_pattern='(\w{4}_\d+)_complex', coreset_index_pattern='(\w{4})_complex')
    
    if run_screening_power:
        prepare_screening(model, screening_paths, output_screening_dir, 
                          batch_size, score_column, 
                          tfrecords_pattern="feature_screening_(\w{4})(_\d+)?.tfrecords", 
                          index_pattern='\w{4}_(\w{4}_ligand_\d+)_complex')
    del model 
    return None





