import sys
sys.path.append('/home/open-share/shiota/aqdnet_20221006/script/')
import os
import glob
import tempfile
import json
from argparse import ArgumentParser
from argparse import RawDescriptionHelpFormatter
from argparse import RawTextHelpFormatter
import pandas as pd
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import aqdnet
from model import ModelByTensorflow
from structure import ElementwiseDNN

MODEL_CLASSES = {
    'ElementwiseDNN': ElementwiseDNN,
}

def parse_params_json(json_file):
    with open(json_file) as f:
        params = json.load(f)
    return params

def get_model_class(model_class_name):
    try:
        model_class = MODEL_CLASSES[model_class_name]
    except:
        raise ValueError(f"model_class argument must be one of  {list(MODEL_CLASSES.keys())} .")
    return model_class

def generate_feature(input_dir, fg_params, num_cpu=1):
    pdb_files = sorted(glob.glob(os.path.join(input_dir, '*.pdb')))
    with tempfile.NamedTemporaryFile(prefix='input_', suffix='.dat') as temp_file:
        with open(temp_file.name, mode='w') as f:
            f.write('\n'.join(pdb_files))
        fg = aqdnet.FeatureGenerator(**fg_params)
        dataset = fg.generate(temp_file.name, mode='complex', num_cpu=num_cpu)
    return dataset

def main(input_dir, output, model_dir_path, num_cpu, cuda):
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda
    model_path = os.path.join(model_dir_path, 'best_model.h5')
    params_file = os.path.join(model_dir_path, 'params_for_predict.json')
    
    params = parse_params_json(params_file)
    model_class_name = params['model_class_name']
    fg_params = params['fg_params']
    model_params = params['model_params']
    
    model_class = get_model_class(model_class_name)
    
    dataset = generate_feature(input_dir, fg_params, num_cpu)
    model = ModelByTensorflow(network_cls=model_class, **model_params)
    model.load_model(model_path)
    preds = model.predict(dataset)

    print(preds)
    preds.to_csv(output)
    print(f"{output} was saved...")
    return None
    
    
if __name__ == '__main__':
    description = """Predict pKa energy from pdb files.

    Examples:
        singularity shell --nv environment/aqdnet_env_latest.sif
        python predict.py -input_dir "input/" -output "output.csv" -model "../Models/Docking_Energy30RMSD2.5/" -num_cpu 2 -cuda '-1'
    
    """

    parser = ArgumentParser(
        description=description, 
        formatter_class=RawTextHelpFormatter# RawDescriptionHelpFormatter
    )

    parser.add_argument("-input_dir", type=str, default=".",
                        help="""Directory name of input data. Files with a .pdb extension in input_dir are passed to glob.glob() function.
                        """)
    
    parser.add_argument("-output", type=str, default="output.csv",
                        help="""Output file name.
Column Discription:
    pKa_energy: pKa_energy predicted by both complex feature and ligand feature.
    pKa_energy_by_cmp: pKa_energy predicted by only complex feature.
    ligand_energy_by_lgd: Ligand_energy predicted by only ligand feature.
                        """)
    
    parser.add_argument("-model",'--model_dir_path', type=str, default='./',
                        help="""Trained model path. Make a prediction with the model specified by this argument. 
In this model directory, 'params_for_prediction.json' must be placed. 
                        """)
    
    parser.add_argument("-num_cpu", type=int, default=2,
                        help="""Number of CPUs to use in feature generation phase.
                        """)
    parser.add_argument("-cuda",'--cuda_visible_device',  type=str, default="-1",
                        help="""CUDA_VISIBLE_DEVICE argument. If you use GPU devices, set this argument to like '0,1'. Default to '-1'.
                        """)  

    args = parser.parse_args()
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(0)
    
    main(input_dir=args.input_dir, 
         output=args.output, 
         model_dir_path=args.model_dir_path, 
         num_cpu=args.num_cpu, 
         cuda=args.cuda_visible_device
    )
