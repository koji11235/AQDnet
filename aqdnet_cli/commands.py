"""
AQDnet CLI commands implementation
"""

import sys
import os
import glob
import tempfile
import json
import logging

# Add scripts to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

import pandas as pd
import numpy as np
import tensorflow as tf
from tqdm import tqdm

# Import from Scripts
import aqdnet
from model import ModelByTensorflow
from structure import ElementwiseDNN

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MODEL_CLASSES = {
    'ElementwiseDNN': ElementwiseDNN,
}


def discover_pdb_files(input_path):
    """Find PDB files under a directory tree or from a single file path."""
    if os.path.isfile(input_path):
        if not input_path.lower().endswith('.pdb'):
            raise ValueError(f"Input file is not a PDB file: {input_path}")
        return [input_path]

    if not os.path.isdir(input_path):
        raise FileNotFoundError(f"Input path not found: {input_path}")

    return sorted(glob.glob(os.path.join(input_path, '**', '*.pdb'), recursive=True))


def parse_params_json(json_file):
    """Load model parameters from JSON file"""
    with open(json_file) as f:
        params = json.load(f)
    return params


def resolve_model_paths(model_dir):
    """Resolve required model files from a model directory."""
    model_path = os.path.join(model_dir, 'best_model.h5')
    params_file = os.path.join(model_dir, 'params_for_predict.json')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(params_file):
        raise FileNotFoundError(f"Params file not found: {params_file}")
    return model_path, params_file


def load_feature_csv(csv_file):
    """Load a feature CSV and preserve serialized index columns when present."""
    dataset = pd.read_csv(csv_file)
    if 'input_path' in dataset.columns:
        logging.info("Using input_path column from features as index")
        dataset = dataset.set_index('input_path')
    unnamed_columns = [col for col in dataset.columns if str(col).startswith('Unnamed:')]
    if unnamed_columns:
        index_column = unnamed_columns[0]
        logging.info(f"Using index-like column from features: {index_column}")
        dataset = dataset.set_index(index_column)
        if len(unnamed_columns) > 1:
            dataset = dataset.drop(columns=unnamed_columns[1:])
    return dataset


def generate_feature_dataset(input_path, fg_params, num_cpu):
    """Generate features from PDB inputs using the provided feature-generator params."""
    pdb_files = discover_pdb_files(input_path)
    if not pdb_files:
        raise ValueError(f"No PDB files found under {input_path}")

    logging.info(f"Found {len(pdb_files)} PDB files")
    with tempfile.NamedTemporaryFile(prefix='input_', suffix='.dat', delete=False) as temp_file:
        temp_name = temp_file.name
        with open(temp_name, mode='w') as f:
            f.write('\n'.join(pdb_files))

    try:
        fg = aqdnet.FeatureGenerator(**fg_params)
        logging.info("Generating features...")
        dataset = fg.generate(temp_name, mode='complex', num_cpu=num_cpu)
    finally:
        if os.path.exists(temp_name):
            os.remove(temp_name)

    return dataset


def get_model_class(model_class_name):
    """Get model class by name"""
    if model_class_name not in MODEL_CLASSES:
        raise ValueError(f"model_class must be one of {list(MODEL_CLASSES.keys())}")
    return MODEL_CLASSES[model_class_name]


def featurize_command(args):
    """
    Feature extraction command
    """
    logging.info(f"Feature extraction from: {args.input}")
    logging.info(f"Output: {args.output}")

    fg_params = {'lig_code': args.ligand_code}
    if args.feature_param_file:
        if not os.path.exists(args.feature_param_file):
            raise FileNotFoundError(f"Feature param file not found: {args.feature_param_file}")
        fg_params = parse_params_json(args.feature_param_file)['fg_params']
        logging.info(f"Using feature-generation params from: {args.feature_param_file}")

    dataset = generate_feature_dataset(args.input, fg_params, args.num_cpu)
    dataset.to_csv(args.output, index=True, index_label='input_path')
    logging.info(f"Features saved to {args.output}")


def train_command(args):
    """
    Model training command.

    This command validates the requested inputs and points users to the
    notebook-based workflow, which remains the canonical training path.
    """
    logging.info(f"Training model with features: {args.features}")
    logging.info(f"Labels: {args.labels}")
    logging.info(f"Output directory: {args.output_dir}")

    if not os.path.exists(args.features):
        raise FileNotFoundError(f"Feature input not found: {args.features}")
    if not os.path.exists(args.labels):
        raise FileNotFoundError(f"Label file not found: {args.labels}")

    os.makedirs(args.output_dir, exist_ok=True)

    logging.warning(
        "Training remains notebook-first in this repository. "
        "Use 02_train_model.ipynb for the full workflow."
    )
    logging.warning(
        "This CLI command currently performs input validation only and does not "
        "run model fitting end-to-end."
    )


def predict_command(args):
    """
    Prediction command
    """
    logging.info(f"Loading model from: {args.model}")
    logging.info(f"Input: {args.features}")
    logging.info(f"Output: {args.output}")
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    
    model_path, params_file = resolve_model_paths(args.model)
    
    # Parse parameters
    params = parse_params_json(params_file)
    model_class_name = params['model_class_name']
    model_params = params['model_params']
    
    model_class = get_model_class(model_class_name)
    
    # Load or generate features
    if os.path.isdir(args.features) or (
        os.path.isfile(args.features) and args.features.lower().endswith('.pdb')
    ):
        fg_params = params['fg_params']
        logging.info("Generating features from PDB input...")
        dataset = generate_feature_dataset(args.features, fg_params, args.num_cpu)
    elif os.path.isfile(args.features):
        logging.info("Loading features from CSV...")
        dataset = load_feature_csv(args.features)
    else:
        raise FileNotFoundError(f"Feature input not found: {args.features}")
    
    # Make predictions
    logging.info("Loading model...")
    model = ModelByTensorflow(network_cls=model_class, **model_params)
    model.load_model(model_path)
    
    logging.info("Making predictions...")
    preds = model.predict(dataset)

    if isinstance(preds.index, pd.RangeIndex) and not isinstance(dataset.index, pd.RangeIndex):
        preds.index = dataset.index

    index_label = preds.index.name or 'input_path'
    preds.to_csv(args.output, index=True, index_label=index_label)
    logging.info(f"Predictions saved to {args.output}")


def demo_command(args):
    """
    Minimal demo with sample structures
    """
    logging.info("Running minimal demo with sample structures...")
    
    sample_dir = os.path.join(os.path.dirname(__file__), '..', 'sample_structures')
    output_dir = '/tmp/aqdnet_demo'
    os.makedirs(output_dir, exist_ok=True)
    
    # Feature extraction from sample structures
    pdb_files = sorted(glob.glob(os.path.join(sample_dir, '**', '*.pdb'), recursive=True))[:2]
    
    if not pdb_files:
        raise FileNotFoundError(f"No PDB files found in {sample_dir}")
    
    logging.info(f"Demo: Generating features from {len(pdb_files)} sample structures...")
    
    with tempfile.NamedTemporaryFile(prefix='demo_', suffix='.dat', delete=False) as temp_file:
        temp_name = temp_file.name
        with open(temp_name, mode='w') as f:
            f.write('\n'.join(pdb_files))
        
        try:
            fg = aqdnet.FeatureGenerator(lig_code='LGD')
            dataset = fg.generate(temp_name, mode='complex', num_cpu=2)
            
            output_file = os.path.join(output_dir, 'demo_features.csv')
            dataset.to_csv(output_file, index=False)
            logging.info(f"✓ Demo: Features extracted successfully!")
            logging.info(f"  Output: {output_file}")
            logging.info(f"  Shape: {dataset.shape}")
        except Exception as e:
            logging.error(f"Demo feature extraction failed: {e}")
            raise
        finally:
            if os.path.exists(temp_name):
                os.remove(temp_name)
