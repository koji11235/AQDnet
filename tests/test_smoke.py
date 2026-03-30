"""
AQDnet Smoke Tests

Minimal test suite for AQDnet functionality:
- Feature extraction from sample structures
- Model parameter loading and prediction pipeline
- CLI interface availability
"""

import os
import sys
import tempfile
import glob
import json
import shutil
import subprocess
import unittest
from types import SimpleNamespace
from unittest import mock
import pandas as pd
import numpy as np

# Add scripts to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

import aqdnet
from model import ModelByTensorflow
from structure import ElementwiseDNN
from aqdnet_cli import commands


class TestFeatureExtraction(unittest.TestCase):
    """Test feature extraction pipeline"""
    
    def test_feature_generator_initialization(self):
        """Test that FeatureGenerator can be instantiated"""
        fg = aqdnet.FeatureGenerator(lig_code='LGD')
        self.assertIsNotNone(fg)
        self.assertEqual(fg.lig_code, 'LGD')
    
    def test_feature_extraction_from_sample(self):
        """Test feature extraction from sample structures"""
        sample_dir = os.path.join(
            os.path.dirname(__file__), 
            '..', 
            'sample_structures'
        )
        
        # Get PDB files
        pdb_files = sorted(glob.glob(os.path.join(sample_dir, '**', '*.pdb'), recursive=True))[:1]
        
        self.assertGreater(len(pdb_files), 0, f"No PDB files found in {sample_dir}")
        
        # Generate features
        with tempfile.NamedTemporaryFile(prefix='test_', suffix='.dat', delete=False) as temp_file:
            temp_name = temp_file.name
            with open(temp_name, mode='w') as f:
                f.write('\n'.join(pdb_files))
            
            try:
                fg = aqdnet.FeatureGenerator(lig_code='LGD')
                dataset = fg.generate(temp_name, mode='complex', num_cpu=1)
                
                # Verify output
                self.assertIsNotNone(dataset)
                self.assertGreater(len(dataset), 0, "Generated dataset is empty")
                self.assertGreater(dataset.shape[1], 0, "Generated dataset has no features")
                
            finally:
                if os.path.exists(temp_name):
                    os.remove(temp_name)

    def test_recursive_pdb_discovery_from_sample_root(self):
        """Test that CLI PDB discovery finds nested sample structure files."""
        sample_dir = os.path.join(os.path.dirname(__file__), '..', 'sample_structures')
        pdb_files = commands.discover_pdb_files(sample_dir)

        self.assertGreater(len(pdb_files), 0, f"No PDB files found in {sample_dir}")
        self.assertTrue(all(path.endswith('.pdb') for path in pdb_files))


class TestModelPipeline(unittest.TestCase):
    """Test model loading and prediction pipeline"""

    def _model_dir(self, name):
        return os.path.join(os.path.dirname(__file__), '..', 'models', name)

    def _has_real_model_artifact(self, name):
        model_dir = self._model_dir(name)
        return (
            os.path.exists(os.path.join(model_dir, 'params_for_predict.json')) and
            os.path.exists(os.path.join(model_dir, 'best_model.h5'))
        )

    def _generate_feature_csv_for_prediction(self, temp_dir, fg_params):
        sample_dir = os.path.join(
            os.path.dirname(__file__),
            '..',
            'sample_structures',
            'predict_example',
        )
        pdb_files = sorted(glob.glob(os.path.join(sample_dir, '*.pdb')))[:1]
        if not pdb_files:
            self.skipTest("No sample structures available to generate prediction features")

        input_list = os.path.join(temp_dir, 'predict_inputs.dat')
        with open(input_list, 'w') as f:
            f.write('\n'.join(pdb_files))

        fg = aqdnet.FeatureGenerator(**fg_params)
        dataset = fg.generate(input_list, mode='complex', num_cpu=1)

        feature_csv = os.path.join(temp_dir, 'generated_predict_features.csv')
        dataset.to_csv(feature_csv, index=False)
        return feature_csv
    
    def test_model_params_loading(self):
        """Test that model parameters can be loaded"""
        models_dir = os.path.join(
            os.path.dirname(__file__),
            '..',
            'models'
        )
        
        # Check Docking model params
        docking_params_file = os.path.join(
            models_dir,
            'Docking_Energy30RMSD2.5',
            'params_for_predict.json'
        )
        
        self.assertTrue(os.path.exists(docking_params_file), 
                       f"Model params not found: {docking_params_file}")
        
        # Load and verify params structure
        with open(docking_params_file, 'r') as f:
            params = json.load(f)
        
        self.assertIn('model_class_name', params, "Missing model_class_name in params")
        self.assertIn('fg_params', params, "Missing fg_params in params")
        self.assertIn('model_params', params, "Missing model_params in params")
        self.assertEqual(params['model_class_name'], 'ElementwiseDNN', 
                        "Expected ElementwiseDNN model class")
    
    def test_mock_model_params_loading(self):
        """Test loading mock model parameters for testing"""
        mock_model_dir = os.path.join(
            os.path.dirname(__file__),
            'mock_model'
        )
        
        params_file = os.path.join(mock_model_dir, 'params_for_predict.json')
        
        self.assertTrue(os.path.exists(params_file), 
                       f"Mock model params not found: {params_file}")
        
        # Load and verify params
        with open(params_file, 'r') as f:
            params = json.load(f)
        
        self.assertIn('model_class_name', params)
        self.assertIn('fg_params', params)
        self.assertIn('model_params', params)

    def test_predict_with_mock_model_when_real_weights_are_missing(self):
        """Fall back to a mocked prediction path when real model weights are unavailable."""
        if self._has_real_model_artifact('Docking_Energy30RMSD2.5'):
            self.skipTest("Real docking model artifacts are available; mock fallback is not needed.")

        with tempfile.TemporaryDirectory(prefix='aqdnet_mock_predict_') as temp_dir:
            model_dir = os.path.join(temp_dir, 'mock_model')
            os.makedirs(model_dir)
            shutil.copyfile(
                os.path.join(os.path.dirname(__file__), 'mock_model', 'params_for_predict.json'),
                os.path.join(model_dir, 'params_for_predict.json'),
            )

            model_file = os.path.join(model_dir, 'best_model.h5')
            with open(model_file, 'w') as f:
                f.write('mock model placeholder')

            features_file = os.path.join(temp_dir, 'features.csv')
            pd.DataFrame({'f0': [0.1], 'f1': [0.2]}).to_csv(features_file, index=False)

            output_file = os.path.join(temp_dir, 'predictions.csv')
            args = SimpleNamespace(
                model=model_dir,
                features=features_file,
                output=output_file,
                cuda='-1',
            )

            mock_prediction = pd.DataFrame({'prediction': [0.42]})
            with mock.patch.object(commands, 'parse_params_json') as parse_mock, \
                 mock.patch.object(commands, 'get_model_class', return_value=ElementwiseDNN), \
                 mock.patch.object(commands, 'ModelByTensorflow') as model_cls:
                parse_mock.return_value = {
                    'model_class_name': 'ElementwiseDNN',
                    'model_params': {},
                }
                model_instance = model_cls.return_value
                model_instance.predict.return_value = mock_prediction

                commands.predict_command(args)

            self.assertTrue(os.path.exists(output_file))
            preds = pd.read_csv(output_file)
            self.assertEqual(preds.shape, (1, 2))
            self.assertIn('prediction', preds.columns)
            self.assertAlmostEqual(preds['prediction'].iloc[0], 0.42)
            model_instance.load_model.assert_called_once_with(model_file)

    def test_predict_with_real_docking_model_if_available(self):
        """Run the prediction command with the docking model when weights are present."""
        self._run_real_model_prediction_if_available('Docking_Energy30RMSD2.5')

    def _run_real_model_prediction_if_available(self, model_name):
        model_dir = self._model_dir(model_name)
        params_file = os.path.join(model_dir, 'params_for_predict.json')
        model_file = os.path.join(model_dir, 'best_model.h5')
        if not os.path.exists(params_file) or not os.path.exists(model_file):
            self.skipTest(f"Real model artifacts not available for {model_name}")

        with tempfile.TemporaryDirectory(prefix='aqdnet_real_predict_') as temp_dir:
            params = commands.parse_params_json(params_file)
            feature_csv = self._generate_feature_csv_for_prediction(
                temp_dir,
                params['fg_params'],
            )

            output_file = os.path.join(temp_dir, f'{model_name}_predictions.csv')
            args = SimpleNamespace(
                model=model_dir,
                features=feature_csv,
                output=output_file,
                cuda='-1',
            )

            commands.predict_command(args)

            self.assertTrue(os.path.exists(output_file))
            preds = pd.read_csv(output_file)
            self.assertGreater(len(preds), 0)

    def test_model_class_instantiation(self):
        """Test that ElementwiseDNN model class can be instantiated with default params"""
        # Use minimal parameters - ElementwiseDNN has many but mostly have defaults
        try:
            model = ModelByTensorflow(network_cls=ElementwiseDNN)
            self.assertIsNotNone(model)
        except Exception as e:
            self.fail(f"Failed to instantiate ModelByTensorflow: {e}")


class TestCLIInterface(unittest.TestCase):
    """Test CLI interface availability"""
    
    def test_cli_import(self):
        """Test that CLI module can be imported"""
        try:
            import aqdnet_cli
            self.assertIsNotNone(aqdnet_cli)
        except ImportError:
            self.skipTest("aqdnet_cli not installed in test environment")
    
    def test_cli_commands_available(self):
        """Test that CLI commands are properly defined"""
        self.assertTrue(hasattr(commands, 'featurize_command'))
        self.assertTrue(hasattr(commands, 'predict_command'))
        self.assertTrue(hasattr(commands, 'demo_command'))

    def test_console_entry_point_exports_main(self):
        """Test that the package exports the main CLI function."""
        import aqdnet_cli

        self.assertTrue(hasattr(aqdnet_cli, 'main'))
        self.assertTrue(callable(aqdnet_cli.main))

    def test_cli_featurize_subprocess_on_nested_input(self):
        """Test the CLI entry point against a nested sample input directory."""
        sample_root = os.path.join(os.path.dirname(__file__), '..', 'sample_structures')
        source_pdb = sorted(glob.glob(os.path.join(sample_root, '**', '*.pdb'), recursive=True))[0]

        with tempfile.TemporaryDirectory(prefix='aqdnet_cli_test_') as temp_dir:
            nested_dir = os.path.join(temp_dir, 'nested', 'sample')
            os.makedirs(nested_dir)
            copied_pdb = os.path.join(nested_dir, os.path.basename(source_pdb))
            shutil.copyfile(source_pdb, copied_pdb)

            output_csv = os.path.join(temp_dir, 'features.csv')
            result = subprocess.run(
                [
                    sys.executable,
                    '-m',
                    'aqdnet_cli',
                    'featurize',
                    '--input',
                    temp_dir,
                    '--output',
                    output_csv,
                    '--ligand-code',
                    'LGD',
                    '--num-cpu',
                    '1',
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertTrue(os.path.exists(output_csv), msg=result.stderr)


if __name__ == '__main__':
    unittest.main(verbosity=2)
