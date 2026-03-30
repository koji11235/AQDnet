"""
AQDnet CLI main entry point
"""

import sys
import argparse
from .commands import featurize_command, train_command, predict_command, demo_command


def main():
    parser = argparse.ArgumentParser(
        description="AQDnet: Deep Neural Network for Protein-Ligand Docking and Scoring",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m aqdnet_cli featurize --input sample_structures/ --output features.csv
  python -m aqdnet_cli featurize --input sample_structures/predict_example --feature-param-file models/Docking_Energy30RMSD2.5/params_for_predict.json --output features.csv
  python -m aqdnet_cli predict --model models/Docking_Energy30RMSD2.5 --features sample_structures/predict_example --output predictions.csv
  python -m aqdnet_cli demo
        """
    )
    
    parser.add_argument('--version', action='version', version='%(prog)s 0.1.0')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    subparsers.required = True
    
    # Featurize command
    featurize_parser = subparsers.add_parser('featurize', help='Extract features from PDB structures')
    featurize_parser.add_argument('--input', required=True, type=str, 
                                  help='Input directory containing PDB files')
    featurize_parser.add_argument('--output', required=True, type=str, 
                                  help='Output CSV file for features')
    featurize_parser.add_argument('--ligand-code', type=str, default='LGD',
                                  help='Ligand code in PDB (default: LGD)')
    featurize_parser.add_argument('--feature-param-file', type=str, default=None,
                                  help='Optional params_for_predict.json path; if set, use its fg_params for feature generation')
    featurize_parser.add_argument('--num-cpu', type=int, default=2,
                                  help='Number of CPUs for parallel processing (default: 2)')
    featurize_parser.set_defaults(func=featurize_command)
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train AQDnet model')
    train_parser.add_argument('--features', required=True, type=str,
                              help='Feature CSV file(s) or pattern')
    train_parser.add_argument('--labels', required=True, type=str,
                              help='Labels CSV file')
    train_parser.add_argument('--output-dir', required=True, type=str,
                              help='Output directory for trained model')
    train_parser.add_argument('--epochs', type=int, default=100,
                              help='Number of training epochs (default: 100)')
    train_parser.add_argument('--batch-size', type=int, default=32,
                              help='Batch size (default: 32)')
    train_parser.set_defaults(func=train_command)
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Make predictions with trained model')
    predict_parser.add_argument('--model', required=True, type=str,
                                help='Trained model directory (must contain best_model.h5 and params_for_predict.json)')
    predict_parser.add_argument('--features', required=True, type=str,
                                help='Feature CSV file, PDB directory, or single PDB file')
    predict_parser.add_argument('--output', required=True, type=str,
                                help='Output CSV file for predictions')
    predict_parser.add_argument('--num-cpu', type=int, default=2,
                                help='Number of CPUs (default: 2)')
    predict_parser.add_argument('--cuda', type=str, default='-1',
                                help='CUDA device ID, -1 for CPU (default: -1)')
    predict_parser.set_defaults(func=predict_command)
    
    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Run minimal demo with sample structures')
    demo_parser.set_defaults(func=demo_command)
    
    args = parser.parse_args()
    
    try:
        args.func(args)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
