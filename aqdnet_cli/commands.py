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


def parse_params_json(json_file):
    """Load model parameters from JSON file"""
    with open(json_file) as f:
        params = json.load(f)
    return params


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
    
    if not os.path.isdir(args.input):
        raise FileNotFoundError(f"Input directory not found: {args.input}")
    
    pdb_files = sorted(glob.glob(os.path.join(args.input, '*.pdb')))
    
    if not pdb_files:
        raise ValueError(f"No PDB files found in {args.input}")
    
    logging.info(f"Found {len(pdb_files)} PDB files")
    
    # Generate features
    with tempfile.NamedTemporaryFile(prefix='input_', suffix='.dat', delete=False) as temp_file:
        temp_name = temp_file.name
        with open(temp_name, mode='w') as f:
            f.write('\n'.join(pdb_files))
        
        try:
            fg = aqdnet.FeatureGenerator(lig_code=args.ligand_code)
            logging.info("Generating features...")
            dataset = fg.generate(temp_name, mode='complex', num_cpu=args.num_cpu)
            
            # Save to CSV
            dataset.to_csv(args.output, index=False)
            logging.info(f"Features saved to {args.output}")
        finally:
            if os.path.exists(temp_name):
                os.remove(temp_name)


def train_command(args):
    """
    Model training command
    
    Note: This is a placeholder. Full training requires more setup.
    """
    logging.info(f"Training model with features: {args.features}")
    logging.info(f"Labels: {args.labels}")
    logging.info(f"Output directory: {args.output_dir}")
    
    raise NotImplementedError(
        "Training CLI is not fully implemented yet. "
        "Please use Ex2_train_model.ipynb for complete training workflow."
    )


def predict_command(args):
    """
    Prediction command
    """
    logging.info(f"Loading model from: {args.model}")
    logging.info(f"Input features: {args.features}")
    logging.info(f"Output: {args.output}")
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    
    # Check model files
    model_path = os.path.join(args.model, 'best_model.h5')
    params_file = os.path.join(args.model, 'params_for_predict.json')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(params_file):
        raise FileNotFoundError(f"Params file not found: {params_file}")
    
    # Parse parameters
    params = parse_params_json(params_file)
    model_class_name = params['model_class_name']
    model_params = params['model_params']
    
    model_class = get_model_class(model_class_name)
    
    # Load features
    if os.path.isfile(args.features):
        logging.info("Loading features from CSV...")
        dataset = pd.read_csv(args.features)
    else:
        raise FileNotFoundError(f"Feature file not found: {args.features}")
    
    # Make predictions
    logging.info("Loading model...")
    model = ModelByTensorflow(network_cls=model_class, **model_params)
    model.load_model(model_path)
    
    logging.info("Making predictions...")
    preds = model.predict(dataset)
    
    preds.to_csv(args.output, index=False)
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
