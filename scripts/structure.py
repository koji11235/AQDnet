from abc import ABCMeta
from abc import abstractmethod
import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm
import itertools

from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import Add

from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow_addons.layers import SpectralNormalization
from tensorflow.data import TFRecordDataset
from tensorflow.python.data.ops.dataset_ops import BatchDataset

import tensorflow as tf

import runner


class MyAbstractNetworkStructure(metaclass=ABCMeta):
    """Meta class of Network structures below. 
    NetworkStructure class specifies the structure of the neural network and its input data preprocessing.

    Args:
        metaclass ([type], optional): [description]. Defaults to ABCMeta.
    """

    def __init__(self):
        return None

    @abstractmethod
    def create(self):
        pass

    @abstractmethod
    def preprocess(self):
        pass

    # calculate_metrics was removed. Please use self.model.evaluate instead.
    # @abstractmethod
    # def calculate_metrics(self): # evaluateとかのほうがいい?
    #     pass

##################################################################################
###################          custom loss functions             ###################
##################################################################################


def _PCC_RMSE(y_true, y_pred, alpha=0.7):
    """Hybrid loss function of pearson correlation coefficient (PCC) and root mean squared error (RMSE).
    ref: https://github.com/zchwang/OnionNet-2/blob/master/train.py

    Args:
        y_true : Ground truth values. shape = `[batch_size, d0, .. dN]`.
        y_pred : The predicted values. shape = `[batch_size, d0, .. dN]`
        alpha (float, optional): Ratio of PCC and RMSE. If it is set to 0, return PCC loss. If it is set to 1, return RMSE loss. Float in [0, 1]. Defaults to 0.7.

    Returns:
        [type]: [description]
    """
    fsp = y_pred - tf.keras.backend.mean(y_pred)
    fst = y_true - tf.keras.backend.mean(y_true)

    devP = tf.keras.backend.std(y_pred)
    devT = tf.keras.backend.std(y_true)

    rmse = tf.keras.backend.sqrt(tf.keras.backend.mean(
        tf.keras.backend.square(y_pred - y_true), axis=-1))

    pcc = 1.0 - tf.keras.backend.mean(fsp * fst) / (devP * devT)
    loss = alpha * pcc + (1 - alpha) * rmse

    return loss


def make_PCC_RMSE_with_alpha(alpha):
    """generate PCC_RMSE loss function with different alpha value.

    Args:
        alpha ([float]): Ratio of PCC and RMSE. If it is set to 0, return PCC loss. If it is set to 1, return RMSE loss. Float in [0, 1]. Defaults to 0.7.

    Returns:
        [function]: PCC_RMSE loss with specified alpha.
    """
    def PCC_RMSE(y_true, y_pred):
        fsp = y_pred - tf.keras.backend.mean(y_pred)
        fst = y_true - tf.keras.backend.mean(y_true)

        devP = tf.keras.backend.std(y_pred)
        devT = tf.keras.backend.std(y_true)

        rmse = tf.keras.backend.sqrt(tf.keras.backend.mean(
            tf.keras.backend.square(y_pred - y_true), axis=-1))

        pcc = 1.0 - tf.keras.backend.mean(fsp * fst) / (devP * devT)
        loss = alpha * pcc + (1 - alpha) * rmse

        return loss
    return PCC_RMSE


def RMSE(y_true, y_pred):
    """Root mean squared error loss function.

    Args:
        y_true : Ground truth values. shape = `[batch_size, d0, .. dN]`.
        y_pred : The predicted values. shape = `[batch_size, d0, .. dN]`

    Returns:
        [type]: Root mean squared error values.
    """
    return tf.keras.backend.sqrt(tf.keras.backend.mean(tf.keras.backend.square(y_pred - y_true), axis=-1))


def PCC(y_true, y_pred):
    """Pearson correlation coefficient loss function.

    Args:
        y_true : Ground truth values.
        y_pred : The predicted values.

    Returns:
        [type]: Pearson correlation coefficient value.
    """
    fsp = y_pred - tf.keras.backend.mean(y_pred)
    fst = y_true - tf.keras.backend.mean(y_true)

    devP = tf.keras.backend.std(y_pred)
    devT = tf.keras.backend.std(y_true)

    pcc = tf.keras.backend.mean(fsp * fst) / (devP * devT)
    pcc = tf.where(tf.math.is_nan(pcc), 0.8, pcc)
    return pcc


#####################################################################################################
###################     Network Structure classes (Now Developping)      ###################
#####################################################################################################
class ElementwiseDNN(MyAbstractNetworkStructure):
    """A neural network that separates features for each elemental species of the ligand and learns with a model for each elemental species. 
    Used as network_cls argument of ModelNN_tensorflow class.
    This network structure class is highly dependent on the dimensionality and names of the feature (Especially in the pre-processing stage). 
    Therefore, if you changed the configurations of FeatureGenerator, please change the Feature-related model_params (input_size_per_element, target_elements, n_target_elements, n_radial_Rs, n_angular_Rs and n_thetas).
    CAUTION: Setting clipnorm or clipvalue is currently unsupported when using a distribution strategy (MirroredStrategy).

    Args:
        MyAbstractNetworkStructure ([type]): MyAbstractNetworkStructure class above
    """

    def __init__(self, label_colnames=['pKa_energy'],
                 # Ver1 26048/8  # 32768/8 + 512/8  # prev ver: 576, # 4096/8 + 512/8,
                 input_size_per_element=1287,
                 input_size_per_element_radial=207,
                 input_size_per_element_angular=1080,
                 n_nodes=[500]*6,
                 output_size_per_element=10,
                 use_residual_dense=True, n_layers_per_res_dense=1,
                 output_layer_style='fc', output_layer_n_nodes=[256]*3,
                 use_spectral_norm=False, l2_norm=0.1, dropout=0.15, dropout_input=0.05,
                 lr=0.001, clipnorm=1, pcc_rmse_alpha=0.7,
                 target_elements=['H', 'C', 'N', 'O', 'P', 'S', 'Cl', 'Zn', 'DU'],
                 n_radial_Rs=23, n_angular_Rs=3, n_thetas=8,
                 separate_radial_and_angular_feature=True,
                 additional_feature_dim=0,
                 use_only_radial_feature=False):
        """Instantiate a model with the specified parameters

        Args:
            label_colnames (list, optional): Column names of the target label. Defaults to ['pKa_energy'].
            model_params (optional): This can accept the choices below.
                input_size_per_element (int): Default to 4160 (32768/8 + 512/8).
                target_elements (list): Element types to be considered. Default to ['H', 'C', 'N', 'O', 'P', 'S', 'Cl', 'DU'].
                n_target_elements (int): Number of element types to be considered. Default to 8.
                n_radial_Rs (int): Number of Rs args in radial symmetry function. Default to 8.
                n_angular_Rs (int): Number of Rs args in angular symmetry function. Default to 8.
                n_thetas (int): Number of theta args in angular symmetry function. Default to 8.
                n_nodes (list): Number of nodes per layer.
                lr (float): Learning rate. Default to 0.0001.
                clipnorm (float): Clip norm value. Default to 1. *Setting clipnorm or clipvalue is currently unsupported when using a distribution strategy (MirroredStrategy).
                output_layer_style (str): Style for summarizing the output from the each element type networks . 'sum', 'fc' or 'sum+fc' is acceptable. Default to 'sum'.
                use_spectral_norm (bool): Whethere to use spectral normalization to Dense layers. Default to True.
                pcc_rmse_alpha (float): Ratio of PCC to RMSE in the loss function. Default to 0.7.
                additional_feature_dim: Number of the feature dimension 

        Returns:
            [type]: [description]
        """
        n_target_elements = len(target_elements)
        radial_feature_dim = n_radial_Rs*(n_target_elements**2)
        angular_feature_dim = int(
            n_angular_Rs*n_thetas*(n_target_elements**3 + n_target_elements**2)/2)

        # if separate_radial_and_angular_feature:
        #     input_size_per_element_radial = int(radial_feature_dim / n_target_elements)
        #     input_size_per_element_angular = int(angular_feature_dim / n_target_elements)

        self.label_colnames = label_colnames
        self.model_params = dict(
            input_size_per_element=input_size_per_element,
            input_size_per_element_radial=input_size_per_element_radial,
            input_size_per_element_angular=input_size_per_element_angular,
            n_nodes=n_nodes,
            output_size_per_element=output_size_per_element,
            use_residual_dense=use_residual_dense,
            n_layers_per_res_dense=n_layers_per_res_dense,
            output_layer_style=output_layer_style,
            output_layer_n_nodes=output_layer_n_nodes,
            use_spectral_norm=use_spectral_norm,
            l2_norm=l2_norm,
            dropout=dropout,
            dropout_input=dropout_input,
            lr=lr,
            clipnorm=clipnorm,
            pcc_rmse_alpha=pcc_rmse_alpha,
        )
        self.feature_params = dict(
            target_elements=target_elements,
            n_target_elements=n_target_elements,
            n_radial_Rs=n_radial_Rs,
            n_angular_Rs=n_angular_Rs,
            n_thetas=n_thetas,
            radial_feature_dim=radial_feature_dim,
            angular_feature_dim=angular_feature_dim,
            feature_dim=radial_feature_dim + angular_feature_dim,
            additional_feature_dim=additional_feature_dim,
        )
        self.separate_radial_and_angular_feature = separate_radial_and_angular_feature
        self.use_additional_feature = additional_feature_dim >= 1
        self.use_only_radial_feature = use_only_radial_feature
        return None

    # , target_elements, n_radial_Rs, n_angular_Rs, n_thetas):
    def create_feature_name(self):
        """Create feature names from given parameters. Feature names are required for preprocessing.

        Args:
            self ([type]): [description]

        Returns:
            feature name [list]: List of radial feature name and angular feature names
        """
        target_elements = self.feature_params['target_elements']
        n_radial_Rs = self.feature_params['n_radial_Rs']
        n_angular_Rs = self.feature_params['n_angular_Rs']
        n_thetas = self.feature_params['n_thetas']

        radial_feature_name = [f"{e_l}_{e_r}_{rs}"
                               for e_l, e_r in itertools.product(target_elements, repeat=2)
                               for rs in range(n_radial_Rs)
                               ]

        target_elements_triplet = [f"{e_j}_{e_i}_{e_k}" for (e_i, (e_j, e_k)) in
                                   list(itertools.product(
                                        target_elements,
                                        list(itertools.combinations_with_replacement(
                                            target_elements, 2))
                                        ))
                                   ]
        angular_feature_name = [f"{element_triplet}_{theta}_{rs}"
                                for element_triplet in target_elements_triplet
                                for theta in range(n_thetas)
                                for rs in range(n_angular_Rs)
                                ]

        # radial_feature_name = [f"{e_l}_{e_r}_{rs}"
        #     for e_l, e_r in itertools.product(target_elements, repeat=2)
        #         for rs in range(n_radial_Rs)
        # ]
        # angular_feature_name = [f"{e_j}_{e_i}_{e_k}_{theta}_{rs}"
        #     for e_j, e_i, e_k in itertools.product(target_elements, repeat=3)
        #         for theta in range(n_thetas)
        #             for rs in range(n_angular_Rs)
        # ]
        return radial_feature_name + angular_feature_name

    def create_dense_block(self, input_size, n_node, use_spectral_norm=False, l2_norm=0.01, dropout=0.5):
        input = Input(shape=input_size)

        if use_spectral_norm:
            x = SpectralNormalization(
                Dense(n_node,
                      kernel_regularizer=l2(l2_norm),
                      activation='relu'),
                dynamic=True
            )(input)
        else:
            x = Dense(n_node,
                      kernel_regularizer=l2(l2_norm),
                      activation='relu'
                      )(input)

        x = BatchNormalization()(x)
        output = Dropout(dropout)(x)
        model = tf.keras.models.Model(
            inputs=[input],
            outputs=[output],
            # name=name
        )
        return model

    def create_residual_dense_block(self, n_node=100, n_layers=1,
                                    use_spectral_norm=False, l2_norm=0.01, dropout=0.2):
        input = Input(shape=n_node)
        if use_spectral_norm:
            x = SpectralNormalization(
                Dense(n_node,
                      kernel_regularizer=l2(l2_norm),
                      activation='relu'),
                dynamic=True
            )(input)
        else:
            x = Dense(n_node,
                      kernel_regularizer=l2(l2_norm),
                      activation='relu'
                      )(input)
        x = BatchNormalization()(x)
        x = Dropout(dropout)(x)

        for _ in range(n_layers-1):
            if use_spectral_norm:
                x = SpectralNormalization(
                    Dense(n_node,
                          kernel_regularizer=l2(l2_norm),
                          activation='relu'),
                    dynamic=True
                )(x)
            else:
                x = Dense(n_node,
                          kernel_regularizer=l2(l2_norm),
                          activation='relu'
                          )(x)
            x = BatchNormalization()(x)
            x = Dropout(dropout)(x)
        output = Add()([x, input])
        model = tf.keras.models.Model(
            inputs=[input],
            outputs=[output],
            # name=name
        )
        return model

    def create_variable_network(self, n_nodes, input_size=4160, output_size=1,
                                use_residual_dense=True, n_layers_per_res_dense=1,
                                use_spectral_norm=False, l2_norm=0.01, dropout=0.5, dropout_input=0.2):
        if use_residual_dense and not len(set(n_nodes)) == 1:
            # check all the values in n_nodes same.
            raise ValueError(
                "If use_residual_dense is True, all the values in n_nodes must be same.")

        input = Input(shape=input_size)
        x = Dropout(dropout_input)(input)
        x = self.create_dense_block(
            input_size=input_size, n_node=n_nodes[0],
            use_spectral_norm=use_spectral_norm,
            l2_norm=l2_norm, dropout=dropout
        )(x)
        prev_n_node = n_nodes[0]
        for n_node in n_nodes[1:]:
            if use_residual_dense:
                x = self.create_residual_dense_block(
                    n_node=n_node, n_layers=n_layers_per_res_dense,
                    use_spectral_norm=use_spectral_norm,
                    l2_norm=l2_norm, dropout=dropout
                )(x)
            else:
                x = self.create_dense_block(
                    input_size=prev_n_node, n_node=n_node,
                    use_spectral_norm=use_spectral_norm,
                    l2_norm=l2_norm, dropout=dropout
                )(x)
            prev_n_node = n_node

        if use_spectral_norm:
            output = SpectralNormalization(Dense(output_size), dynamic=True)(x)
        else:
            output = Dense(output_size)(x)

        model = tf.keras.models.Model(
            inputs=[input],
            outputs=[output],
            # name=name
        )
        return model

    def create_elementwise_dnn(self, n_target_elements=8, input_size_per_element=4160,
                               n_nodes=[64, 64], output_size_per_element=1,
                               use_residual_dense=True, n_layers_per_res_dense=1,
                               use_spectral_norm=True, l2_norm=0.01, dropout=0.5, dropout_input=0.2,
                               name='elementwise_dnn'):
        """Create tensorflow model. 

        Raises:
            ValueError: Raised if output_layer_style is not sum, fc or sum+fc.

        Returns:
            network [tensorflow.python.keras.engine.training.Model]: A model that has networks for each element type.
        """
        # set params -------------------------------------------------------
        # n_target_elements = self.feature_params['n_target_elements']
        # input_size_per_element = self.model_params['input_size_per_element']
        # n_nodes = self.model_params['n_nodes']
        # use_spectral_norm = self.model_params['use_spectral_norm']
        # output_size_per_element = self.model_params['output_size_per_element']
        # l2_norm = self.model_params['l2_norm']
        # dropout = self.model_params['dropout']
        # dropout_input = self.model_params['dropout_input']

        # create middle layers -------------------------------------------------------
        inputs = [Input(shape=input_size_per_element)
                  for _ in range(n_target_elements)]

        elementwise_models = [
            self.create_variable_network(
                n_nodes,
                input_size=input_size_per_element,
                output_size=output_size_per_element,
                use_residual_dense=use_residual_dense,
                n_layers_per_res_dense=n_layers_per_res_dense,
                use_spectral_norm=use_spectral_norm,
                l2_norm=l2_norm,
                dropout=dropout,
                dropout_input=dropout_input
            )
            for _ in range(n_target_elements)
        ]

        outputs = [model(input)
                   for model, input in zip(elementwise_models, inputs)]

        # instantiate the model -------------------------------------------------------
        network = tf.keras.models.Model(
            inputs=[inputs],
            outputs=outputs,
            name=name
        )
        return network

    def create_output_layer(self, output_layer_style, n_nodes=[256],
                            use_spectral_norm=False, l2_norm=0.01, dropout=0.5,
                            name='output', n_inputs=None, input_size=None):

        inputs = [Input(shape=input_size) for _ in range(n_inputs)]

        if output_layer_style == 'sum':
            output = tf.keras.layers.Add()(inputs)

        elif output_layer_style == 'fc':
            x = concatenate(inputs)
            x = Dropout(dropout)(x)

            if l2_norm != 0.0:
                for n_node in n_nodes:
                    x = SpectralNormalization(Dense(n_node, kernel_regularizer=l2(l2_norm), activation='relu'), dynamic=True)(x) \
                        if use_spectral_norm else Dense(n_node, kernel_regularizer=l2(l2_norm), activation='relu')(x)
                    x = Dropout(dropout)(x)
            elif l2_norm == 0.0:
                for n_node in n_nodes:
                    x = SpectralNormalization(Dense(n_node, activation='relu'), dynamic=True)(x) \
                        if use_spectral_norm else Dense(n_node, activation='relu')(x)
                    x = Dropout(dropout)(x)

            output = SpectralNormalization(Dense(1), dynamic=True)(x) \
                if use_spectral_norm else Dense(1)(x)

        elif output_layer_style == 'sum+fc':
            x = tf.keras.layers.Add()(inputs)
            if l2_norm != 0.0:
                for n_node in n_nodes:
                    x = SpectralNormalization(Dense(n_node, kernel_regularizer=l2(l2_norm), activation='relu'), dynamic=True)(x) \
                        if use_spectral_norm else Dense(n_node, kernel_regularizer=l2(l2_norm), activation='relu')(x)
                    x = Dropout(dropout)(x)
            elif l2_norm == 0.0:
                for n_node in n_nodes:
                    x = SpectralNormalization(Dense(n_node, activation='relu'), dynamic=True)(x) \
                        if use_spectral_norm else Dense(n_node, activation='relu')(x)
                    x = Dropout(dropout)(x)

            output = SpectralNormalization(Dense(1), dynamic=True)(x)\
                if use_spectral_norm else Dense(1)(x)

        else:
            raise ValueError('output_layer_style must be sum, fc or sum+fc')

        network = tf.keras.models.Model(
            inputs=[inputs],
            outputs=[output],
            name=name
        )
        return network

    def create_no_separate_mode_model(self, n_target_elements=8, input_size_per_element=4160,
                                      n_nodes=[64, 64], output_size_per_element=1,
                                      output_layer_style='fc', output_layer_n_nodes=[256],
                                      use_residual_dense=True, n_layers_per_res_dense=1,
                                      use_spectral_norm=True, l2_norm=0.01, dropout=0.5, dropout_input=0.2,
                                      use_additional_feature=False, additional_feature_dim=1):
        # create network -------------------------------------------------------
        elementwise_dnn = self.create_elementwise_dnn(
            n_target_elements=n_target_elements,
            input_size_per_element=input_size_per_element,
            n_nodes=n_nodes,
            output_size_per_element=output_size_per_element,
            use_residual_dense=use_residual_dense,
            n_layers_per_res_dense=n_layers_per_res_dense,
            use_spectral_norm=use_spectral_norm,
            l2_norm=l2_norm,
            dropout=dropout,
            dropout_input=dropout_input,
            name='elementwise_dnn'
        )

        if not use_additional_feature:
            output_layer = self.create_output_layer(
                output_layer_style,
                n_nodes=output_layer_n_nodes,
                n_inputs=n_target_elements,
                input_size=output_size_per_element,
                use_spectral_norm=use_spectral_norm,
                l2_norm=l2_norm,
                dropout=dropout,
                name='output'
            )
            inputs = [Input(shape=input_size_per_element)
                      for _ in range(n_target_elements)]
            x = elementwise_dnn(inputs)
            output = output_layer(x)
        else:
            output_layer = self.create_output_layer(
                output_layer_style,
                n_nodes=output_layer_n_nodes,
                n_inputs=n_target_elements + additional_feature_dim,
                input_size=output_size_per_element,
                use_spectral_norm=use_spectral_norm,
                l2_norm=l2_norm,
                dropout=dropout,
                name='output'
            )
            inputs = [Input(shape=input_size_per_element)
                      for _ in range(n_target_elements)]
            additional_inputs = Input(shape=additional_feature_dim)
            x = elementwise_dnn(inputs)
            x = x + [additional_inputs]
            output = output_layer(x)

        # instantiate the model and compile -------------------------------------------------------
        if not use_additional_feature:
            model = tf.keras.models.Model(
                inputs=[inputs],
                outputs=[output]
            )
        else:
            model = tf.keras.models.Model(
                inputs=[inputs, additional_inputs],
                outputs=[output]
            )
        return model

    def create_separate_mode_model(self, n_target_elements=9,
                                   input_size_per_element_radial=99, input_size_per_element_angular=3960,
                                   n_nodes=[64, 64],
                                   output_size_per_element=1,
                                   output_layer_style='fc', output_layer_n_nodes=[256],
                                   use_residual_dense=True, n_layers_per_res_dense=1,
                                   use_spectral_norm=True, l2_norm=0.01, dropout=0.5, dropout_input=0.2,
                                   use_additional_feature=False, additional_feature_dim=1):
        # create network -------------------------------------------------------
        elementwise_dnn_radial = self.create_elementwise_dnn(
            n_target_elements=n_target_elements,
            input_size_per_element=input_size_per_element_radial,
            n_nodes=n_nodes,
            output_size_per_element=output_size_per_element,
            use_residual_dense=use_residual_dense,
            n_layers_per_res_dense=n_layers_per_res_dense,
            use_spectral_norm=use_spectral_norm,
            l2_norm=l2_norm,
            dropout=dropout,
            dropout_input=dropout_input,
            name='elementwise_dnn_radial'
        )

        elementwise_dnn_angular = self.create_elementwise_dnn(
            n_target_elements=n_target_elements,
            input_size_per_element=input_size_per_element_angular,
            n_nodes=n_nodes,
            output_size_per_element=output_size_per_element,
            use_residual_dense=use_residual_dense,
            n_layers_per_res_dense=n_layers_per_res_dense,
            use_spectral_norm=use_spectral_norm,
            l2_norm=l2_norm,
            dropout=dropout,
            dropout_input=dropout_input,
            name='elementwise_dnn_angular'
        )

        if not use_additional_feature:
            output_layer = self.create_output_layer(
                output_layer_style,
                n_nodes=output_layer_n_nodes,
                n_inputs=n_target_elements * 2,
                input_size=output_size_per_element,
                use_spectral_norm=use_spectral_norm,
                l2_norm=l2_norm,
                dropout=dropout,
                name='output'
            )
            inputs_radial = [Input(shape=input_size_per_element_radial)
                             for _ in range(n_target_elements)]
            inputs_angular = [Input(shape=input_size_per_element_angular)
                              for _ in range(n_target_elements)]
            x_radial = elementwise_dnn_radial(inputs_radial)
            x_angular = elementwise_dnn_angular(inputs_angular)
            x = x_radial + x_angular
            output = output_layer(x)
        else:
            output_layer = self.create_output_layer(
                output_layer_style,
                n_nodes=output_layer_n_nodes,
                n_inputs=n_target_elements + additional_feature_dim,
                input_size=output_size_per_element,
                use_spectral_norm=use_spectral_norm,
                l2_norm=l2_norm,
                dropout=dropout,
                name='output'
            )
            inputs_radial = [Input(shape=input_size_per_element_radial)
                             for _ in range(n_target_elements)]
            inputs_angular = [Input(shape=input_size_per_element_angular)
                              for _ in range(n_target_elements)]
            additional_inputs = Input(shape=additional_feature_dim)
            x_radial = elementwise_dnn_radial(inputs_radial)
            x_angular = elementwise_dnn_angular(inputs_angular)
            x = x_radial + x_angular + [additional_inputs]
            output = output_layer(x)

        # instantiate the model and compile -------------------------------------------------------
        if not use_additional_feature:
            model = tf.keras.models.Model(
                inputs=[inputs_radial, inputs_angular],
                outputs=[output]
            )
        else:
            model = tf.keras.models.Model(
                inputs=[inputs_radial, inputs_angular, additional_inputs],
                outputs=[output]
            )
        return model

    def create(self):
        """Create tensorflow model. 

        Raises:
            ValueError: Raised if output_layer_style is not sum, fc or sum+fc.

        Returns:
            network [tensorflow.python.keras.engine.training.Model]: A model that has networks for each element type.
        """
        # set params -------------------------------------------------------
        n_target_elements = self.feature_params['n_target_elements']
        input_size_per_element = self.model_params['input_size_per_element']
        n_nodes = self.model_params['n_nodes']
        output_size_per_element = self.model_params['output_size_per_element']
        output_layer_style = self.model_params['output_layer_style']
        output_layer_n_nodes = self.model_params['output_layer_n_nodes']
        use_residual_dense = self.model_params['use_residual_dense']
        n_layers_per_res_dense = self.model_params['n_layers_per_res_dense']
        use_spectral_norm = self.model_params['use_spectral_norm']
        l2_norm = self.model_params['l2_norm']
        dropout = self.model_params['dropout']
        dropout_input = self.model_params['dropout_input']
        use_additional_feature = self.use_additional_feature
        additional_feature_dim = self.feature_params['additional_feature_dim']
        separate_radial_and_angular_feature = self.separate_radial_and_angular_feature
        input_size_per_element_radial = self.model_params['input_size_per_element_radial']
        input_size_per_element_angular = self.model_params['input_size_per_element_angular']

        if not separate_radial_and_angular_feature:
            model = self.create_no_separate_mode_model(n_target_elements, input_size_per_element,
                                                       n_nodes, output_size_per_element,
                                                       output_layer_style, output_layer_n_nodes,
                                                       use_residual_dense, n_layers_per_res_dense,
                                                       use_spectral_norm, l2_norm, dropout, dropout_input,
                                                       use_additional_feature, additional_feature_dim)
        else:
            model = self.create_separate_mode_model(n_target_elements,
                                                    input_size_per_element_radial, input_size_per_element_angular,
                                                    n_nodes,output_size_per_element,
                                                    output_layer_style, output_layer_n_nodes,
                                                    use_residual_dense, n_layers_per_res_dense,
                                                    use_spectral_norm, l2_norm, dropout, dropout_input,
                                                    use_additional_feature, additional_feature_dim)

        model = self.compile(model)
        return model

    def compile(self, model):
        """function for compiling the network.

        Args:
            network ([type]): [description]

        Returns:
            [type]: [description]
        """
        clipnorm = self.model_params['clipnorm']
        pcc_rmse_alpha = self.model_params['pcc_rmse_alpha']
        lr = self.model_params['lr']

        optimizer = SGD(
            lr=lr,
            momentum=0.9,
            # decay=1e-6,
            # clipnorm=clipnorm
        )

        loss_pcc_rmse = make_PCC_RMSE_with_alpha(alpha=pcc_rmse_alpha)

        model.compile(
            loss=loss_pcc_rmse,
            optimizer=optimizer,
            metrics=[RMSE, PCC, loss_pcc_rmse]
        )

        return model

    def _preprocess_pandas(self, X, y=None):
        """Function to preprocess input of data type Pandas.

        Args:
            X ([pandas.core.frame.DataFrame]): Input data. 
            y ([pandas.core.frame.DataFrame], optional): Target data. Defaults to None.

        Returns:
            _X [pandas.core.frame.DataFrame]: Pre-processed input data. 
            _y [pandas.core.frame.DataFrame]: Pre-processed target data. 
        """
        feature_names = self.create_feature_name()
        pf = self.PreprocessFunction(feature_names,
                                     self.feature_params['target_elements'],
                                     additional_feature_dim=self.feature_params['additional_feature_dim'],
                                     use_only_radial_feature=self.use_only_radial_feature,
                                     separate_radial_and_angular_feature=self.separate_radial_and_angular_feature)
        processed_feature, processed_label = pf.process_pandas(X, y)
        return processed_feature, processed_label

    def _preprocess_tfrecord(self, tfrecords):
        """Function to preprocess input of data type TFRecords.

        Args:
            tfrecords ([TFRecordDatasetV2]): Input and target data. 

        Returns:
            [TFRecordDatasetV2]: Pre-processed input and target data.
        """
        feature_names = self.create_feature_name()
        pf = self.PreprocessFunction(feature_names,
                                     self.feature_params['target_elements'],
                                     additional_feature_dim=self.feature_params['additional_feature_dim'],
                                     use_only_radial_feature=self.use_only_radial_feature,
                                     separate_radial_and_angular_feature=self.separate_radial_and_angular_feature)
        # print(type(feature_names))
        # print(type(pf.feature_names))
        # print(pf.feature_names)
        return tfrecords.map(pf.parse_example).map(pf.process_tfrecord)

    def _preprocess_batched_tfrecord(self, batched_tfrecord):
        """Function to preprocess input of data type TFRecords.

        Args:
            tfrecords ([TFRecordDatasetV2]): Input and target data. 

        Returns:
            [TFRecordDatasetV2]: Pre-processed input and target data.
        """
        feature_names = self.create_feature_name()
        pf = self.PreprocessFunction(feature_names,
                                     self.feature_params['target_elements'],
                                     additional_feature_dim=self.feature_params['additional_feature_dim'],
                                     use_only_radial_feature=self.use_only_radial_feature,
                                     separate_radial_and_angular_feature=self.separate_radial_and_angular_feature)
        # print(type(feature_names))
        # print(type(pf.feature_names))
        # print(pf.feature_names)
        return batched_tfrecord.map(pf.parse_batched_example).map(pf.process_batched_tfrecord)

    def preprocess(self, X, y=None):
        """Pre-process input and target data into a form acceptable to the model.

        Args:
            X : Input data. Pandas DataFrame of TFRecordDatasetV2 can be accepted.
            y ([type], optional): Target data. If the type of X is TFRecordDatasetV2 then y arg is ignored. Defaults to None.

        Returns:
            [type]: Pre-processed input and target data
        """
        if isinstance(X, BatchDataset):
            return self._preprocess_batched_tfrecord(X)
        elif isinstance(X, TFRecordDataset):
            return self._preprocess_tfrecord(X)
        elif isinstance(X, pd.core.frame.DataFrame):
            return self._preprocess_pandas(X, y)
        else:
            TypeError('''Feature(X) and label(y) must be pandas.DataFrame or TFRecordDataset.
            Please check the type of your data.''')

    class PreprocessFunction(object):
        """Class for creating functions for preprocessing input and target data.
        Specializes in preprocessing ElementwiseDNN class input data.
        In the ElementwiseDNN class, the feature names are very important because the feature partitioning depends on the names
        This class determines the element of the ligand from the feature name using regular expressions and divide the feature by the element of the ligand.

        Args:
            object ([type]): [description]
        """

        def __init__(self, feature_names,
                     target_elements=['H', 'C', 'N',
                                      'O', 'P', 'S', 'Cl', 'DU'],
                     additional_feature_dim=0,
                     use_only_radial_feature=False,
                     separate_radial_and_angular_feature=False):
            """instantiate by passing feature name. 
            In the ElementwiseDNN class, the feature names are very important because the feature partitioning depends on the names.
            This class determines the element of the ligand from the feature name using regular expressions and divide the feature by the element of the ligand.
            Therefore, this class needs feature_names, target_elements and regex_template.

            Args:
                feature_names ([list, pd.core.strings.StringMethods]): feature names.
                target_elements ([list]): Element types to be considered. Default to ['H', 'C', 'N', 'O', 'P', 'S', 'Cl', 'DU'].
            """
            self.feature_names = pd.Index(feature_names).str \
                if not isinstance(feature_names, pd.core.strings.StringMethods) else feature_names
            self.feature_dim = self.feature_names.__dict__['_orig'].size
            # ['H', 'C', 'N', 'O', 'P', 'S', 'Cl', 'DU']
            self.target_elements = target_elements

            if use_only_radial_feature:
                self.regex_template = "(^{element}_\w?\w?_\d+$)"
            else:
                self.regex_template = "(^\w?\w?_{element}_\w?\w?_\d+_\d+$)|(^{element}_\w?\w?_\d+$)"

            if not separate_radial_and_angular_feature:
                self.feature_masks_of_each_element = [
                    self.feature_names.match(
                        self.regex_template.format(element=element))
                    for element in self.target_elements
                ]
            else:
                self.regex_template_radial = "(^{element}_\w?\w?_\d+$)"
                self.regex_template_angular = "(^\w?\w?_{element}_\w?\w?_\d+_\d+$)"
                self.feature_masks_of_each_element_radial = [
                    self.feature_names.match(
                        self.regex_template_radial.format(element=element))
                    for element in self.target_elements
                ]
                self.feature_masks_of_each_element_angular = [
                    self.feature_names.match(
                        self.regex_template_angular.format(element=element))
                    for element in self.target_elements
                ]
                self.feature_masks_of_each_element = self.feature_masks_of_each_element_radial\
                    + self.feature_masks_of_each_element_angular

            self.additional_feature_dim = additional_feature_dim
            self.use_additional_feature = additional_feature_dim >= 1
            self.use_only_radial_feature = use_only_radial_feature
            self.separate_radial_and_angular_feature = separate_radial_and_angular_feature
            return None

        def process_pandas(self, feature, label=None):
            """Preprocess feature and label of data type pandas DataFrame.

            Args:
                feature ([pandas.core.frame.DataFrame]): Input data.
                label ([pandas.core.frame.DataFrame], optional): Target data. Defaults to None.

            Returns:
                processed_feature [pandas.core.frame.DataFrame]: Processed input data.
                processed_label [pandas.core.frame.DataFrame]: Processed target data.
            """
            processed_feature = [
                feature.loc[:, feature_mask].astype("float64")
                for feature_mask in self.feature_masks_of_each_element
            ]
            processed_label = label.values[:, 0].astype(
                "float64").copy() if label is not None else None
            return processed_feature, processed_label

        # @staticmethod
        def parse_example(self, example):
            """parse tfrecord to feature and label.

            Args:
                example ([tensorflow.python.framework.ops.Tensor]): [description]

            Returns:
                x [tensorflow.python.framework.ops.Tensor]: [description]
                y [tensorflow.python.framework.ops.Tensor]: [description]
            """
            # feature_dim = 33280
            # features = tf.io.parse_single_example(
            #     example,
            #     features={
            #         'feature': tf.io.FixedLenFeature([self.feature_dim], dtype=tf.float32),
            #         'label': tf.io.FixedLenFeature([1], dtype=tf.float32),
            #         'sample_id': tf.io.FixedLenFeature([1], dtype=tf.string)
            #     }
            # )
            # feature = features["feature"]
            # label = features["label"]
            # sample_id = features["sample_id"]
            features = tf.io.parse_single_example(
                example,
                features={
                    # Don't set 'feature' in dict key. Unexplained error occurs.
                    'complex_feature': tf.io.FixedLenFeature([self.feature_dim + self.additional_feature_dim],
                                                             dtype=tf.float32),
                    'label': tf.io.FixedLenFeature([1], dtype=tf.float32),
                    'sample_id': tf.io.FixedLenFeature([1], dtype=tf.string)
                }
            )
            complex_feature = features["complex_feature"]
            label = features["label"]
            sample_id = features["sample_id"]

            return complex_feature, label, sample_id

        @tf.autograph.experimental.do_not_convert
        def parse_batched_example(self, example):

            features = tf.io.parse_example(
                example,
                features={
                    # Don't set 'feature' in dict key. Unexplained error occurs.
                    'complex_feature': tf.io.FixedLenFeature([self.feature_dim + self.additional_feature_dim],
                                                             dtype=tf.float32),
                    'label': tf.io.FixedLenFeature([1], dtype=tf.float32),
                    'sample_id': tf.io.FixedLenFeature([1], dtype=tf.string)
                }
            )
            complex_feature = features["complex_feature"]
            label = features["label"]
            sample_id = features["sample_id"]

            return complex_feature, label, sample_id

        @tf.autograph.experimental.do_not_convert
        def process_tfrecord(self, feature, label, sample_id):
            """Preprocess feature and label parsed from example by parse_example function above.

            Args:
                feature ([type]): [description]
                label ([type]): [description]

            Returns:
                [type]: [description]
            """
            processed_features = tuple([
                feature[feature_mask]  # .astype("float64")
                for feature_mask in self.feature_masks_of_each_element
            ])
            return processed_features, label

        @tf.autograph.experimental.do_not_convert
        def process_batched_tfrecord(self, feature, label, sample_id):
            """Preprocess feature and label parsed from example by parse_example function above.

            Args:
                feature ([type]): [description]
                label ([type]): [description]

            Returns:
                [type]: [description]
            """
            if not self.use_additional_feature:
                processed_features = tuple([
                    tf.boolean_mask(feature, feature_mask, axis=1)
                    for feature_mask in self.feature_masks_of_each_element
                ])
            else:
                additional_feature = feature[:, -self.additional_feature_dim:]
                _feature = feature[:, :-self.additional_feature_dim]
                processed_features = tuple([
                    tf.boolean_mask(_feature, feature_mask, axis=1)
                    for feature_mask in self.feature_masks_of_each_element
                ] + [additional_feature])
            return processed_features, label

    def generate_label(self, tfrecords, batch_size=128):
        feature_names = self.create_feature_name()
        pf = self.PreprocessFunction(
            feature_names=feature_names,
            target_elements=self.feature_params['target_elements'],
            additional_feature_dim=self.feature_params['additional_feature_dim'],
        )
        def get_labels(feature, labels, sample_id): return labels
        labels_dataset = tfrecords.batch(batch_size)\
            .map(pf.parse_batched_example)\
            .map(get_labels)
        labels_array = np.concatenate([
            np.hstack([col.numpy().reshape([-1, 1]) for col in label])
            for label in labels_dataset
        ])
        labels_index = self.generate_index(tfrecords)
        labels = self.shape_preds(labels_array, index=labels_index)
        return labels

    def generate_index(self, tfrecords, batch_size=128):
        feature_names = self.create_feature_name()
        pf = self.PreprocessFunction(
            feature_names=feature_names,
            target_elements=self.feature_params['target_elements'],
            additional_feature_dim=self.feature_params['additional_feature_dim'],
        )
        def get_sample_id(feature, label, sample_id): return sample_id
        index_dataset = tfrecords.batch(batch_size)\
            .map(pf.parse_batched_example)\
            .map(get_sample_id)
        index = np.vstack([idx for idx in index_dataset]).flatten()
        index = np.vectorize(lambda bytes_obj: bytes_obj.decode())(index)
        return index

    def shape_preds(self, preds, index):
        """function for formatting prediction results into pandas.DataFrame.

        Args:
            preds ([type]): Values of the result dataframe.
            index ([type]): Index of the result dataframe.

        Returns:
            [type]: Formatted prediction result.
        """
        return pd.DataFrame(preds, columns=self.label_colnames, index=index)

    def compile(self, network):
        """function for compiling the network.

        Args:
            network ([type]): [description]

        Returns:
            [type]: [description]
        """
        optimizer = SGD(
            lr=self.model_params['lr'],
            momentum=0.9,
            decay=1e-6,
            # clipnorm=clipnorm
        )

        # network.compile(
        #     loss={'output': 'mse'},
        #     optimizer=optimizer,
        #     metrics={
        #         'output': ['mae', 'mse'],
        #     }
        # )

        loss_pcc_rmse = make_PCC_RMSE_with_alpha(
            alpha=self.model_params['pcc_rmse_alpha'])

        network.compile(
            loss={'output': loss_pcc_rmse},  # {'output': 'mse'},
            optimizer=optimizer,
            metrics={
                'output': ['mse', RMSE, PCC, loss_pcc_rmse],
            }
        )
        return network
    
    class TfrecordsWriter:
        def __init__(self):
            return None
        
        @staticmethod
        def _float_feature(value):
            """return float_list from float / double """
            return tf.train.Feature(float_list=tf.train.FloatList(value=value))

        @staticmethod
        def _bytes_feature(value):
            """return byte_list from string / byte """
            if isinstance(value, type(tf.constant(0))):
                # BytesList won't unpack a string from an EagerTensor.
                value = value.numpy()
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

        @staticmethod
        def record2example(cmp_feature, label, sample_id):
            return tf.train.Example(features=tf.train.Features(feature={
                'complex_feature': ElementwiseDNN.TfrecordsWriter._float_feature(cmp_feature),
                'label': ElementwiseDNN.TfrecordsWriter._float_feature(label),
                'sample_id': ElementwiseDNN.TfrecordsWriter._bytes_feature(sample_id)
            }))

        @staticmethod
        def _write_feature_to_tfrecords(feature_file, label_file,
                                        tfr_filename, feature_dimension=26048,
                                        label_colnames=['pKa_energy'],
                                        dask_sample=400000,
                                        scheduler='processes', 
                                        feature_index_modify=None): # function to modify index # feature_index_modify='_complex
            # load features and labels
            feature_cmp = runner.load_feature(
                feature_file, 
                feature_dimension=feature_dimension, 
                sample=dask_sample, scheduler=scheduler,
            ).astype('float32')
                
            if label_file is not None:
                label = runner.load_label(
                    label_file, label_colnames=label_colnames).astype('float32')
            else:
                label = pd.DataFrame(np.nan, index=feature_cmp.index, columns=['Dummy_Label'])
                
            if feature_index_modify is not None:
                feature_cmp.index = [feature_index_modify(idx) for idx in feature_cmp.index]
            
            # remove not common index
            common_idx = feature_cmp.index.intersection(label.index)
            feature_cmp = feature_cmp.loc[common_idx]
            label = label.loc[common_idx]
            # check feature
            if not all(feature_cmp.index == label.index):
                raise ValueError('feature_cmp and label have different indexes.')
            if len(feature_cmp.index) == 0:
                print('ValueError:', feature_file)
                raise ValueError(
                    f"There are no features to write. Length of feature is 0. file name: {feature_file}")

            # create byte type sample_id
            sample_id = np.vectorize(lambda x: x.encode())(label.index.values)

            # write data to tfr_filename
            n_sample = sample_id.shape[0]
            with tf.io.TFRecordWriter(tfr_filename) as writer:
                for i in tqdm(range(n_sample)):
                    ex = ElementwiseDNN.TfrecordsWriter.record2example(
                        cmp_feature=feature_cmp.values[i],
                        label=label.values[i],
                        sample_id=sample_id[i]
                    )
                    writer.write(ex.SerializeToString())
            return None

        @staticmethod
        def _write_feature_to_tfrecords_with_additional_feature(feature_file, additional_feature_file, 
                                                                label_file, tfr_filename,
                                                                feature_dimension=26048,
                                                                additional_feature_colnames=['energy_kcal/mol_binding'], 
                                                                label_colnames=['pKa_energy'],
                                                                dask_sample=400000, scheduler='processes', 
                                                                feature_index_modify=None):
            # load features and labels
            feature_cmp = runner.load_feature(
                feature_file, 
                feature_dimension=feature_dimension, 
                sample=dask_sample, scheduler=scheduler,
            ).astype('float32')
            label = runner.load_label(
                label_file, label_colnames=label_colnames).astype('float32')
            additional_feature = runner.load_label(additional_feature_file,
                                                label_colnames=additional_feature_colnames).astype('float32')
            # pd.read_csv(additional_feature_file)[additional_feature_colnames]
            feature_cmp = feature_cmp.join(additional_feature)

            # process feature_cmp index
            if feature_index_modify is not None:
                feature_cmp.index = [feature_index_modify(idx) for idx in feature_cmp.index]

            # remove not common index
            common_idx = feature_cmp.index.intersection(label.index)

            feature_cmp = feature_cmp.loc[common_idx]
            label = label.loc[common_idx]

            # check feature
            if not all(feature_cmp.index == label.index):
                raise ValueError('feature_cmp and label have different indexes.')
            if len(feature_cmp.index) == 0:
                raise ValueError(
                    'There are no features to write. Length of feature is 0.')

            # create byte type sample_id
            sample_id = np.vectorize(lambda x: x.encode())(label.index.values)

            # write data to tfr_filename
            n_sample = sample_id.shape[0]
            with tf.io.TFRecordWriter(tfr_filename) as writer:
                for i in tqdm(range(n_sample)):
                    ex = ElementwiseDNN.TfrecordsWriter.record2example(
                        cmp_feature=feature_cmp.values[i],
                        label=label.values[i],
                        sample_id=sample_id[i]
                    )
                    writer.write(ex.SerializeToString())
            return None

        @staticmethod
        def write(feature_file, label_file=None, tfr_filename=None, feature_dimension=26048, 
                  additional_feature_file=None, additional_feature_colnames=None,
                  label_colnames=['pKa_energy'], dask_sample=400000, scheduler='processes',
                  feature_index_modify=None):
            if additional_feature_file is None:
                ElementwiseDNN.TfrecordsWriter._write_feature_to_tfrecords(
                    feature_file, label_file, tfr_filename,
                    feature_dimension=feature_dimension,
                    label_colnames=label_colnames,
                    dask_sample=dask_sample,
                    scheduler=scheduler,
                    feature_index_modify=feature_index_modify,
                )
            else:
                ElementwiseDNN.TfrecordsWriter._write_feature_to_tfrecords_with_additional_feature(
                    feature_file, additional_feature_file, label_file, tfr_filename,
                    feature_dimension=feature_dimension,
                    additional_feature_colnames=additional_feature_colnames,
                    label_colnames=label_colnames,
                    dask_sample=dask_sample,
                    scheduler=scheduler,
                    feature_index_modify=feature_index_modify,
                )
            print(f"{tfr_filename} was saved...")
            return None
    
    @staticmethod
    def write_tfrecords(feature_file, label_file=None, tfr_filename=None, feature_dimension=26048, 
                        additional_feature_file=None, additional_feature_colnames=None,
                        label_colnames=['pKa_energy'], dask_sample=400000, scheduler='processes',
                        feature_index_modify=None):
        ElementwiseDNN.TfrecordsWriter.write(feature_file, label_file=label_file, 
                                             tfr_filename=tfr_filename,
                                             feature_dimension=feature_dimension,
                                             label_colnames=label_colnames,
                                             dask_sample=dask_sample,
                                             scheduler=scheduler,
                                             feature_index_modify=feature_index_modify)
        return None

    


class PrevElementwiseDNN(MyAbstractNetworkStructure):
    """A neural network that separates features for each elemental species of the ligand and learns with a model for each elemental species. 
    Used as network_cls argument of ModelNN_tensorflow class.
    This network structure class is highly dependent on the dimensionality and names of the feature (Especially in the pre-processing stage). 
    Therefore, if you changed the configurations of FeatureGenerator, please change the Feature-related model_params (input_size_per_element, target_elements, n_target_elements, n_radial_Rs, n_angular_Rs and n_thetas).
    CAUTION: Setting clipnorm or clipvalue is currently unsupported when using a distribution strategy (MirroredStrategy).

    Args:
        MyAbstractNetworkStructure ([type]): MyAbstractNetworkStructure class above
    """

    def __init__(self, label_colnames=['pKa_energy'],
                 input_size_per_element=4160,  # 32768/8 + 512/8  # prev ver: 576, # 4096/8 + 512/8,
                 n_nodes=[64, 64], output_size_per_element=1,
                 use_residual_dense=True, n_layers_per_res_dense=1,
                 output_layer_style='fc', output_layer_n_nodes=[256],
                 use_spectral_norm=True, l2_norm=0.01, dropout=0.5, dropout_input=0.2,
                 lr=0.0001, clipnorm=1, pcc_rmse_alpha=0.7,
                 target_elements=['H', 'C', 'N', 'O', 'P', 'S', 'Cl', 'DU'],
                 n_radial_Rs=8, n_angular_Rs=8, n_thetas=8,):
        """Instantiate a model with the specified parameters

        Args:
            label_colnames (list, optional): Column names of the target label. Defaults to ['pKa_energy'].
            model_params (optional): This can accept the choices below.
                input_size_per_element (int): Default to 4160 (32768/8 + 512/8).
                target_elements (list): Element types to be considered. Default to ['H', 'C', 'N', 'O', 'P', 'S', 'Cl', 'DU'].
                n_target_elements (int): Number of element types to be considered. Default to 8.
                n_radial_Rs (int): Number of Rs args in radial symmetry function. Default to 8.
                n_angular_Rs (int): Number of Rs args in angular symmetry function. Default to 8.
                n_thetas (int): Number of theta args in angular symmetry function. Default to 8.
                n_nodes (list): Number of nodes per layer.
                lr (float): Learning rate. Default to 0.0001.
                clipnorm (float): Clip norm value. Default to 1. *Setting clipnorm or clipvalue is currently unsupported when using a distribution strategy (MirroredStrategy).
                output_layer_style (str): Style for summarizing the output from the each element type networks . 'sum', 'fc' or 'sum+fc' is acceptable. Default to 'sum'.
                use_spectral_norm (bool): Whethere to use spectral normalization to Dense layers. Default to True.
                pcc_rmse_alpha (float): Ratio of PCC to RMSE in the loss function. Default to 0.7.

        Returns:
            [type]: [description]
        """
        n_target_elements = len(target_elements)
        radial_feature_dim = n_radial_Rs*(n_target_elements**2)
        angular_feature_dim = n_angular_Rs*n_thetas*(n_target_elements**3)

        self.label_colnames = label_colnames
        self.model_params = dict(
            input_size_per_element=input_size_per_element,
            n_nodes=n_nodes,
            output_size_per_element=output_size_per_element,
            use_residual_dense=use_residual_dense,
            n_layers_per_res_dense=n_layers_per_res_dense,
            output_layer_style=output_layer_style,
            output_layer_n_nodes=output_layer_n_nodes,
            use_spectral_norm=use_spectral_norm,
            l2_norm=l2_norm,
            dropout=dropout,
            dropout_input=dropout_input,
            lr=lr,
            clipnorm=clipnorm,
            pcc_rmse_alpha=pcc_rmse_alpha,
        )
        self.feature_params = dict(
            target_elements=target_elements,
            n_target_elements=n_target_elements,
            n_radial_Rs=n_radial_Rs,
            n_angular_Rs=n_angular_Rs,
            n_thetas=n_thetas,
            radial_feature_dim=radial_feature_dim,
            angular_feature_dim=angular_feature_dim,
            feature_dim=radial_feature_dim + angular_feature_dim,
        )
        return None

    # , target_elements, n_radial_Rs, n_angular_Rs, n_thetas):
    def create_feature_name(self):
        """Create feature names from given parameters. Feature names are required for preprocessing.

        Args:
            self ([type]): [description]

        Returns:
            feature name [list]: List of radial feature name and angular feature names
        """
        target_elements = self.feature_params['target_elements']
        n_radial_Rs = self.feature_params['n_radial_Rs']
        n_angular_Rs = self.feature_params['n_angular_Rs']
        n_thetas = self.feature_params['n_thetas']

        radial_feature_name = [f"{e_l}_{e_r}_{rs}"
                               for e_l, e_r in itertools.product(target_elements, repeat=2)
                               for rs in range(n_radial_Rs)
                               ]
        angular_feature_name = [f"{e_j}_{e_i}_{e_k}_{theta}_{rs}"
                                for e_j, e_i, e_k in itertools.product(target_elements, repeat=3)
                                for theta in range(n_thetas)
                                for rs in range(n_angular_Rs)
                                ]
        return radial_feature_name + angular_feature_name

    def create_dense_block(self, input_size, n_node, use_spectral_norm=False, l2_norm=0.01, dropout=0.5):
        input = Input(shape=input_size)

        if use_spectral_norm:
            x = SpectralNormalization(
                Dense(n_node,
                      kernel_regularizer=l2(l2_norm),
                      activation='relu'),
                dynamic=True
            )(input)
        else:
            x = Dense(n_node,
                      kernel_regularizer=l2(l2_norm),
                      activation='relu'
                      )(input)

        x = BatchNormalization()(x)
        output = Dropout(dropout)(x)
        model = tf.keras.models.Model(
            inputs=[input],
            outputs=[output],
            # name=name
        )
        return model

    def create_residual_dense_block(self, n_node=100, n_layers=1,
                                    use_spectral_norm=False, l2_norm=0.01, dropout=0.2):
        input = Input(shape=n_node)
        if use_spectral_norm:
            x = SpectralNormalization(
                Dense(n_node,
                      kernel_regularizer=l2(l2_norm),
                      activation='relu'),
                dynamic=True
            )(input)
        else:
            x = Dense(n_node,
                      kernel_regularizer=l2(l2_norm),
                      activation='relu'
                      )(input)
        x = BatchNormalization()(x)
        x = Dropout(dropout)(x)

        for _ in range(n_layers-1):
            if use_spectral_norm:
                x = SpectralNormalization(
                    Dense(n_node,
                          kernel_regularizer=l2(l2_norm),
                          activation='relu'),
                    dynamic=True
                )(x)
            else:
                x = Dense(n_node,
                          kernel_regularizer=l2(l2_norm),
                          activation='relu'
                          )(x)
            x = BatchNormalization()(x)
            x = Dropout(dropout)(x)
        output = Add()([x, input])
        model = tf.keras.models.Model(
            inputs=[input],
            outputs=[output],
            # name=name
        )
        return model

    def create_variable_network(self, n_nodes, input_size=4160, output_size=1,
                                use_residual_dense=True, n_layers_per_res_dense=1,
                                use_spectral_norm=False, l2_norm=0.01, dropout=0.5, dropout_input=0.2):
        if use_residual_dense and not len(set(n_nodes)) == 1:
            # check all the values in n_nodes same.
            raise ValueError(
                "If use_residual_dense is True, all the values in n_nodes must be same.")

        input = Input(shape=input_size)
        x = Dropout(dropout_input)(input)
        x = self.create_dense_block(
            input_size=input_size, n_node=n_nodes[0],
            use_spectral_norm=use_spectral_norm,
            l2_norm=l2_norm, dropout=dropout
        )(x)
        prev_n_node = n_nodes[0]
        for n_node in n_nodes[1:]:
            if use_residual_dense:
                x = self.create_residual_dense_block(
                    n_node=n_node, n_layers=n_layers_per_res_dense,
                    use_spectral_norm=use_spectral_norm,
                    l2_norm=l2_norm, dropout=dropout
                )(x)
            else:
                x = self.create_dense_block(
                    input_size=prev_n_node, n_node=n_node,
                    use_spectral_norm=use_spectral_norm,
                    l2_norm=l2_norm, dropout=dropout
                )(x)
            prev_n_node = n_node

        if use_spectral_norm:
            output = SpectralNormalization(Dense(output_size), dynamic=True)(x)
        else:
            output = Dense(output_size)(x)

        model = tf.keras.models.Model(
            inputs=[input],
            outputs=[output],
            # name=name
        )
        return model

    def create_elementwise_dnn(self, n_target_elements=8, input_size_per_element=4160,
                               n_nodes=[64, 64], output_size_per_element=1,
                               use_residual_dense=True, n_layers_per_res_dense=1,
                               use_spectral_norm=True, l2_norm=0.01, dropout=0.5, dropout_input=0.2,
                               name='elementwise_dnn'):
        """Create tensorflow model. 

        Raises:
            ValueError: Raised if output_layer_style is not sum, fc or sum+fc.

        Returns:
            network [tensorflow.python.keras.engine.training.Model]: A model that has networks for each element type.
        """
        # set params -------------------------------------------------------
        # n_target_elements = self.feature_params['n_target_elements']
        # input_size_per_element = self.model_params['input_size_per_element']
        # n_nodes = self.model_params['n_nodes']
        # use_spectral_norm = self.model_params['use_spectral_norm']
        # output_size_per_element = self.model_params['output_size_per_element']
        # l2_norm = self.model_params['l2_norm']
        # dropout = self.model_params['dropout']
        # dropout_input = self.model_params['dropout_input']

        # create middle layers -------------------------------------------------------
        inputs = [Input(shape=input_size_per_element)
                  for _ in range(n_target_elements)]

        elementwise_models = [
            self.create_variable_network(
                n_nodes,
                input_size=input_size_per_element,
                output_size=output_size_per_element,
                use_residual_dense=use_residual_dense,
                n_layers_per_res_dense=n_layers_per_res_dense,
                use_spectral_norm=use_spectral_norm,
                l2_norm=l2_norm,
                dropout=dropout,
                dropout_input=dropout_input
            )
            for _ in range(n_target_elements)
        ]

        outputs = [model(input)
                   for model, input in zip(elementwise_models, inputs)]

        # instantiate the model -------------------------------------------------------
        network = tf.keras.models.Model(
            inputs=[inputs],
            outputs=outputs,
            name=name
        )
        return network

    def create_output_layer(self, output_layer_style, n_nodes=[256],
                            use_spectral_norm=False, l2_norm=0.01, dropout=0.5,
                            name='output', n_inputs=None, input_size=None):

        inputs = [Input(shape=input_size) for _ in range(n_inputs)]

        if output_layer_style == 'sum':
            output = tf.keras.layers.Add()(inputs)

        elif output_layer_style == 'fc':
            x = concatenate(inputs)
            x = Dropout(dropout)(x)

            if l2_norm != 0.0:
                for n_node in n_nodes:
                    x = SpectralNormalization(Dense(n_node, kernel_regularizer=l2(l2_norm), activation='relu'), dynamic=True)(x) \
                        if use_spectral_norm else Dense(n_node, kernel_regularizer=l2(l2_norm), activation='relu')(x)
                    x = Dropout(dropout)(x)
            elif l2_norm == 0.0:
                for n_node in n_nodes:
                    x = SpectralNormalization(Dense(n_node, activation='relu'), dynamic=True)(x) \
                        if use_spectral_norm else Dense(n_node, activation='relu')(x)
                    x = Dropout(dropout)(x)

            output = SpectralNormalization(Dense(1), dynamic=True)(x) \
                if use_spectral_norm else Dense(1)(x)

        elif output_layer_style == 'sum+fc':
            x = tf.keras.layers.Add()(inputs)
            if l2_norm != 0.0:
                for n_node in n_nodes:
                    x = SpectralNormalization(Dense(n_node, kernel_regularizer=l2(l2_norm), activation='relu'), dynamic=True)(x) \
                        if use_spectral_norm else Dense(n_node, kernel_regularizer=l2(l2_norm), activation='relu')(x)
                    x = Dropout(dropout)(x)
            elif l2_norm == 0.0:
                for n_node in n_nodes:
                    x = SpectralNormalization(Dense(n_node, activation='relu'), dynamic=True)(x) \
                        if use_spectral_norm else Dense(n_node, activation='relu')(x)
                    x = Dropout(dropout)(x)

            output = SpectralNormalization(Dense(1), dynamic=True)(x)\
                if use_spectral_norm else Dense(1)(x)

        else:
            raise ValueError('output_layer_style must be sum, fc or sum+fc')

        network = tf.keras.models.Model(
            inputs=[inputs],
            outputs=[output],
            name=name
        )
        return network

    def compile(self, model):
        """function for compiling the network.

        Args:
            network ([type]): [description]

        Returns:
            [type]: [description]
        """
        clipnorm = self.model_params['clipnorm']
        pcc_rmse_alpha = self.model_params['pcc_rmse_alpha']
        lr = self.model_params['lr']

        optimizer = SGD(
            lr=lr,
            momentum=0.9,
            decay=1e-6,
            # clipnorm=clipnorm
        )

        loss_pcc_rmse = make_PCC_RMSE_with_alpha(alpha=pcc_rmse_alpha)

        model.compile(
            loss=loss_pcc_rmse,
            optimizer=optimizer,
            metrics=[RMSE, PCC, loss_pcc_rmse]
        )

        return model

    def create(self):
        """Create tensorflow model. 

        Raises:
            ValueError: Raised if output_layer_style is not sum, fc or sum+fc.

        Returns:
            network [tensorflow.python.keras.engine.training.Model]: A model that has networks for each element type.
        """
        # set params -------------------------------------------------------
        n_target_elements = self.feature_params['n_target_elements']
        input_size_per_element = self.model_params['input_size_per_element']
        n_nodes = self.model_params['n_nodes']
        output_size_per_element = self.model_params['output_size_per_element']
        output_layer_style = self.model_params['output_layer_style']
        output_layer_n_nodes = self.model_params['output_layer_n_nodes']
        use_residual_dense = self.model_params['use_residual_dense']
        n_layers_per_res_dense = self.model_params['n_layers_per_res_dense']
        use_spectral_norm = self.model_params['use_spectral_norm']
        l2_norm = self.model_params['l2_norm']
        dropout = self.model_params['dropout']
        dropout_input = self.model_params['dropout_input']

        # create network -------------------------------------------------------
        elementwise_dnn = self.create_elementwise_dnn(
            n_target_elements=n_target_elements,
            input_size_per_element=input_size_per_element,
            n_nodes=n_nodes,
            output_size_per_element=output_size_per_element,
            use_residual_dense=use_residual_dense,
            n_layers_per_res_dense=n_layers_per_res_dense,
            use_spectral_norm=use_spectral_norm,
            l2_norm=l2_norm,
            dropout=dropout,
            dropout_input=dropout_input,
            name='elementwise_dnn'
        )

        output_layer = self.create_output_layer(
            output_layer_style,
            n_nodes=output_layer_n_nodes,
            n_inputs=n_target_elements,
            input_size=output_size_per_element,
            use_spectral_norm=use_spectral_norm,
            l2_norm=l2_norm,
            dropout=dropout,
            name='output'
        )
        inputs = [Input(shape=input_size_per_element)
                  for _ in range(n_target_elements)]
        x = elementwise_dnn(inputs)
        output = output_layer(x)

        # instantiate the model and compile -------------------------------------------------------
        model = tf.keras.models.Model(
            inputs=[inputs],
            outputs=[output]
        )
        model = self.compile(model)
        return model

    def _preprocess_pandas(self, X, y=None):
        """Function to preprocess input of data type Pandas.

        Args:
            X ([pandas.core.frame.DataFrame]): Input data. 
            y ([pandas.core.frame.DataFrame], optional): Target data. Defaults to None.

        Returns:
            _X [pandas.core.frame.DataFrame]: Pre-processed input data. 
            _y [pandas.core.frame.DataFrame]: Pre-processed target data. 
        """
        feature_names = self.create_feature_name()
        pf = self.PreprocessFunction(feature_names)
        processed_feature, processed_label = pf.process_pandas(X, y)
        return processed_feature, processed_label

    def _preprocess_tfrecord(self, tfrecords):
        """Function to preprocess input of data type TFRecords.

        Args:
            tfrecords ([TFRecordDatasetV2]): Input and target data. 

        Returns:
            [TFRecordDatasetV2]: Pre-processed input and target data.
        """
        feature_names = self.create_feature_name()
        pf = self.PreprocessFunction(
            feature_names, self.feature_params['target_elements'])
        # print(type(feature_names))
        # print(type(pf.feature_names))
        # print(pf.feature_names)
        return tfrecords.map(pf.parse_example).map(pf.process_tfrecord)

    def _preprocess_batched_tfrecord(self, batched_tfrecord):
        """Function to preprocess input of data type TFRecords.

        Args:
            tfrecords ([TFRecordDatasetV2]): Input and target data. 

        Returns:
            [TFRecordDatasetV2]: Pre-processed input and target data.
        """
        feature_names = self.create_feature_name()
        pf = self.PreprocessFunction(
            feature_names, self.feature_params['target_elements'])
        # print(type(feature_names))
        # print(type(pf.feature_names))
        # print(pf.feature_names)
        return batched_tfrecord.map(pf.parse_batched_example).map(pf.process_batched_tfrecord)

    def preprocess(self, X, y=None):
        """Pre-process input and target data into a form acceptable to the model.

        Args:
            X : Input data. Pandas DataFrame of TFRecordDatasetV2 can be accepted.
            y ([type], optional): Target data. If the type of X is TFRecordDatasetV2 then y arg is ignored. Defaults to None.

        Returns:
            [type]: Pre-processed input and target data
        """
        if isinstance(X, BatchDataset):
            return self._preprocess_batched_tfrecord(X)
        elif isinstance(X, TFRecordDataset):
            return self._preprocess_tfrecord(X)
        elif isinstance(X, pd.core.frame.DataFrame):
            return self._preprocess_pandas(X, y)
        else:
            TypeError('''Feature(X) and label(y) must be pandas.DataFrame or TFRecordDataset.
            Please check the type of your data.''')

    class PreprocessFunction(object):
        """Class for creating functions for preprocessing input and target data.
        Specializes in preprocessing ElementwiseDNN class input data.
        In the ElementwiseDNN class, the feature names are very important because the feature partitioning depends on the names
        This class determines the element of the ligand from the feature name using regular expressions and divide the feature by the element of the ligand.

        Args:
            object ([type]): [description]
        """

        def __init__(self, feature_names, target_elements=['H', 'C', 'N', 'O', 'P', 'S', 'Cl', 'DU']):
            """instantiate by passing feature name. 
            In the ElementwiseDNN class, the feature names are very important because the feature partitioning depends on the names.
            This class determines the element of the ligand from the feature name using regular expressions and divide the feature by the element of the ligand.
            Therefore, this class needs feature_names, target_elements and regex_template.

            Args:
                feature_names ([list, pd.core.strings.StringMethods]): feature names.
                target_elements ([list]): Element types to be considered. Default to ['H', 'C', 'N', 'O', 'P', 'S', 'Cl', 'DU'].
            """
            self.feature_names = pd.Index(feature_names).str \
                if not isinstance(feature_names, pd.core.strings.StringMethods) else feature_names
            self.feature_dim = self.feature_names.__dict__['_orig'].size
            # ['H', 'C', 'N', 'O', 'P', 'S', 'Cl', 'DU']
            self.target_elements = target_elements
            self.regex_template = "(^\w?\w?_{element}_\w?\w?_\d+_\d+$)|(^{element}_\w?\w?_\d+$)"
            self.feature_masks_of_each_element = [
                self.feature_names.match(
                    self.regex_template.format(element=element))
                for element in self.target_elements
            ]

        def process_pandas(self, feature, label=None):
            """Preprocess feature and label of data type pandas DataFrame.

            Args:
                feature ([pandas.core.frame.DataFrame]): Input data.
                label ([pandas.core.frame.DataFrame], optional): Target data. Defaults to None.

            Returns:
                processed_feature [pandas.core.frame.DataFrame]: Processed input data.
                processed_label [pandas.core.frame.DataFrame]: Processed target data.
            """
            processed_feature = [
                feature.loc[:, feature_mask].astype("float64")
                for feature_mask in self.feature_masks_of_each_element
            ]
            processed_label = label.values[:, 0].astype(
                "float64").copy() if label is not None else None
            return processed_feature, processed_label

        # @staticmethod
        def parse_example(self, example):
            """parse tfrecord to feature and label.

            Args:
                example ([tensorflow.python.framework.ops.Tensor]): [description]

            Returns:
                x [tensorflow.python.framework.ops.Tensor]: [description]
                y [tensorflow.python.framework.ops.Tensor]: [description]
            """
            # feature_dim = 33280
            # features = tf.io.parse_single_example(
            #     example,
            #     features={
            #         'feature': tf.io.FixedLenFeature([self.feature_dim], dtype=tf.float32),
            #         'label': tf.io.FixedLenFeature([1], dtype=tf.float32),
            #         'sample_id': tf.io.FixedLenFeature([1], dtype=tf.string)
            #     }
            # )
            # feature = features["feature"]
            # label = features["label"]
            # sample_id = features["sample_id"]
            features = tf.io.parse_single_example(
                example,
                features={
                    # Don't set 'feature' in dict key. Unexplained error occurs.
                    'complex_feature': tf.io.FixedLenFeature([self.feature_dim], dtype=tf.float32),
                    'label': tf.io.FixedLenFeature([1], dtype=tf.float32),
                    'sample_id': tf.io.FixedLenFeature([1], dtype=tf.string)
                }
            )
            complex_feature = features["complex_feature"]
            label = features["label"]
            sample_id = features["sample_id"]

            return complex_feature, label, sample_id

        @tf.autograph.experimental.do_not_convert
        def parse_batched_example(self, example):
            features = tf.io.parse_example(
                example,
                features={
                    # Don't set 'feature' in dict key. Unexplained error occurs.
                    'complex_feature': tf.io.FixedLenFeature([self.feature_dim], dtype=tf.float32),
                    'label': tf.io.FixedLenFeature([1], dtype=tf.float32),
                    'sample_id': tf.io.FixedLenFeature([1], dtype=tf.string)
                }
            )
            complex_feature = features["complex_feature"]
            label = features["label"]
            sample_id = features["sample_id"]

            return complex_feature, label, sample_id

        @tf.autograph.experimental.do_not_convert
        def process_tfrecord(self, feature, label, sample_id):
            """Preprocess feature and label parsed from example by parse_example function above.

            Args:
                feature ([type]): [description]
                label ([type]): [description]

            Returns:
                [type]: [description]
            """
            processed_features = tuple([
                feature[feature_mask]  # .astype("float64")
                for feature_mask in self.feature_masks_of_each_element
            ])
            return processed_features, label

        @tf.autograph.experimental.do_not_convert
        def process_batched_tfrecord(self, feature, label, sample_id):
            """Preprocess feature and label parsed from example by parse_example function above.

            Args:
                feature ([type]): [description]
                label ([type]): [description]

            Returns:
                [type]: [description]
            """
            processed_features = tuple([
                tf.boolean_mask(feature, feature_mask, axis=1)
                for feature_mask in self.feature_masks_of_each_element
            ])
            return processed_features, label

    # def generate_label(self, tfrecords, batch_size=128):
    #     labels_dataset = self.preprocess(tfrecords).map(lambda feature, labels: labels).batch(batch_size)
    #     labels_array = np.concatenate([np.hstack([col.numpy().reshape([-1,1]) for col in label]) for label in labels_dataset])
    #     labels_index = self.generate_index(tfrecords)
    #     labels = self.shape_preds(labels_array, index=labels_index)
    #     return labels

    def generate_label(self, tfrecords, batch_size=128):
        feature_names = self.create_feature_name()
        pf = self.PreprocessFunction(
            feature_names=feature_names,
            target_elements=self.feature_params['target_elements']
        )
        def get_labels(feature, labels, sample_id): return labels
        labels_dataset = tfrecords.batch(batch_size)\
            .map(pf.parse_batched_example)\
            .map(get_labels)
        labels_array = np.concatenate([
            np.hstack([col.numpy().reshape([-1, 1]) for col in label])
            for label in labels_dataset
        ])
        labels_index = self.generate_index(tfrecords)
        labels = self.shape_preds(labels_array, index=labels_index)
        return labels

    def generate_index(self, tfrecords, batch_size=128):
        feature_names = self.create_feature_name()
        pf = self.PreprocessFunction(
            feature_names=feature_names,
            target_elements=self.feature_params['target_elements']
        )
        def get_sample_id(feature, label, sample_id): return sample_id
        index_dataset = tfrecords.batch(batch_size)\
            .map(pf.parse_batched_example)\
            .map(get_sample_id)
        index = np.vstack([idx for idx in index_dataset]).flatten()
        index = np.vectorize(lambda bytes_obj: bytes_obj.decode())(index)
        return index

    def shape_preds(self, preds, index):
        """function for formatting prediction results into pandas.DataFrame.

        Args:
            preds ([type]): Values of the result dataframe.
            index ([type]): Index of the result dataframe.

        Returns:
            [type]: Formatted prediction result.
        """
        return pd.DataFrame(preds, columns=self.label_colnames, index=index)

    def compile(self, network):
        """function for compiling the network.

        Args:
            network ([type]): [description]

        Returns:
            [type]: [description]
        """
        optimizer = SGD(
            lr=self.model_params['lr'],
            momentum=0.9,
            decay=1e-6,
            # clipnorm=clipnorm
        )

        # network.compile(
        #     loss={'output': 'mse'},
        #     optimizer=optimizer,
        #     metrics={
        #         'output': ['mae', 'mse'],
        #     }
        # )

        loss_pcc_rmse = make_PCC_RMSE_with_alpha(
            alpha=self.model_params['pcc_rmse_alpha'])

        network.compile(
            loss={'output': loss_pcc_rmse},  # {'output': 'mse'},
            optimizer=optimizer,
            metrics={
                'output': ['mse', RMSE, PCC, loss_pcc_rmse],
            }
        )
        return network


class CNN2D(MyAbstractNetworkStructure):
    def __init__(self, label_colnames=['pKa_energy'],
                 target_elements=['H', 'C', 'N', 'O', 'P', 'S', 'Cl', 'DU'], n_radial_Rs=8, n_angular_Rs=8, n_thetas=8,
                 lr=0.0001, clipnorm=1, pcc_rmse_alpha=0.7,
                 # input_size_radial=(8, 8, 8),
                 n_filters_radial=[32, 64, 128], kernel_size_radial=3, strides_radial=1, padding_method_radial='valid',
                 maxpool_radial=False, pool_size_radial=2, maxpool_strides_radial=2, maxpool_padding_method_radial='same',
                 # input_size_angular=(64, 64, 8),
                 n_filters_angular=[32, 64, 128], kernel_size_angular=4, strides_angular=1, padding_method_angular='valid',
                 maxpool_angular=False, pool_size_angular=2, maxpool_strides_angular=2, maxpool_padding_method_angular='same',
                 n_nodes=[128, 64, 32], l2_norm=0.01, use_spectral_norm=False, dropout=0.5
                 ):
        """[summary]

        Args:
            input_size (tuple, optional): Model Parameter. [description]. Defaults to (33280,1).
            label_colnames (list, optional):  Column names of the target label. Defaults to ['pKa'].
            feature_dim (int, optional): Feature Prameter. Number of the dimension of the feature. Defaults to 33280.
            target_elements (list, optional): Feature Prameter. Element types considered in feature generation step. Defaults to ['H', 'C', 'N', 'O', 'P', 'S', 'Cl', 'DU'].
            n_radial_Rs (int, optional): Feature Prameter. Number of Rs args in radial symmetry function. Defaults to 8.
            n_angular_Rs (int, optional): Feature Prameter. Number of Rs args in angular symmetry function. Defaults to 8.
            n_thetas (int, optional): Feature Prameter. Number of theta args in angular symmetry function.  Defaults to 8.
            lr (float, optional): Model Parameter. [description]. Defaults to 0.0001.
            clipnorm (int, optional): Model Parameter. [description]. Defaults to 1.
            pcc_rmse_alpha (float, optional): Model Parameter. Ratio of PCC to RMSE in the loss function. Defaults to 0.7.
            n_filters (list, optional): Model Parameter (CNN). List of the number of the filters per layer. Defaults to [32, 64, 128].
            kernel_size (int, optional):  Model Parameter (CNN). [description]. Defaults to 4.
            strides (int, optional):  Model Parameter (CNN). [description]. Defaults to 1.
            padding_method (str, optional):  Model Parameter (CNN). [description]. Defaults to 'valid'.
            maxpool (bool, optional):  Model Parameter (CNN). description]. Defaults to False.
            pool_size (int, optional):  Model Parameter (CNN). [description]. Defaults to 2.
            maxpool_strides (int, optional):  Model Parameter (CNN). [description]. Defaults to 2.
            maxpool_padding_method (str, optional):  Model Parameter (CNN). [description]. Defaults to 'same'.
            n_nodes (list, optional):  Model Parameter (DNN). List of the number of the nodes per layer. Defaults to [128, 64, 32].
            l2_norm (float, optional): Model Parameter (DNN). [description]. Defaults to 0.01.
            use_spectral_norm (bool, optional): Model Parameter (DNN). [description]. Defaults to False.
            dropout (float, optional): Model Parameter (DNN). [description]. Defaults to 0.5.

        Returns:
            [type]: [description]
        """
        n_target_elements = len(target_elements)
        radial_feature_dim = n_radial_Rs*(n_target_elements**2)
        angular_feature_dim = n_angular_Rs*n_thetas*(n_target_elements**3)
        input_size_radial = (n_target_elements, n_radial_Rs, n_target_elements)
        input_size_angular = (
            n_target_elements*n_target_elements, n_angular_Rs*n_thetas, n_target_elements)

        self.label_colnames = label_colnames
        self.model_params = dict(
            # input_size=input_size, # 32768/8 + 512/8  # prev ver: 576, # 4096/8 + 512/8,
            lr=lr,
            clipnorm=clipnorm,
            pcc_rmse_alpha=pcc_rmse_alpha,
            radial_cnn_params=dict(
                input_size=input_size_radial,
                n_filters=n_filters_radial,
                kernel_size=kernel_size_radial,
                strides=strides_radial,
                padding_method=padding_method_radial,
                maxpool=maxpool_radial,
                pool_size=pool_size_radial,
                maxpool_strides=maxpool_strides_radial,
                maxpool_padding_method=maxpool_padding_method_radial,
            ),
            angular_cnn_params=dict(
                input_size=input_size_angular,
                n_filters=n_filters_angular,
                kernel_size=kernel_size_angular,
                strides=strides_angular,
                padding_method=padding_method_angular,
                maxpool=maxpool_angular,
                pool_size=pool_size_angular,
                maxpool_strides=maxpool_strides_angular,
                maxpool_padding_method=maxpool_padding_method_angular,
            ),
            dnn_params=dict(
                n_nodes=n_nodes,
                l2_norm=l2_norm,
                use_spectral_norm=use_spectral_norm,
                dropout=dropout
            )
        )
        self.feature_params = dict(
            target_elements=target_elements,
            n_target_elements=n_target_elements,
            n_radial_Rs=n_radial_Rs,
            n_angular_Rs=n_angular_Rs,
            n_thetas=n_thetas,
            radial_feature_dim=radial_feature_dim,
            angular_feature_dim=angular_feature_dim,
            feature_dim=radial_feature_dim + angular_feature_dim,
        )
        return None

    # , target_elements, n_radial_Rs, n_angular_Rs, n_thetas):
    def create_feature_name(self):
        """Create feature names from given parameters. Feature names are required for preprocessing.

        Args:
            self ([type]): [description]

        Returns:
            feature name [list]: List of radial feature name and angular feature names
        """
        target_elements = self.feature_params['target_elements']
        n_radial_Rs = self.feature_params['n_radial_Rs']
        n_angular_Rs = self.feature_params['n_angular_Rs']
        n_thetas = self.feature_params['n_thetas']

        radial_feature_name = [f"{e_l}_{e_r}_{rs}"
                               for e_l, e_r in itertools.product(target_elements, repeat=2)
                               for rs in range(n_radial_Rs)
                               ]
        angular_feature_name = [f"{e_j}_{e_i}_{e_k}_{theta}_{rs}"
                                for e_j, e_i, e_k in itertools.product(target_elements, repeat=3)
                                for theta in range(n_thetas)
                                for rs in range(n_angular_Rs)
                                ]
        return radial_feature_name + angular_feature_name

    def create_cnn2d_layer(self, input_size=(8, 8, 8), n_filters=[32, 64, 128], kernel_size=3, strides=1, padding_method='valid', maxpool=False, pool_size=2, maxpool_strides=2, maxpool_padding_method='same'):
        model = tf.keras.Sequential()
        model.add(Conv2D(n_filters[0], kernel_size=kernel_size,
                  strides=strides, padding=padding_method, input_shape=input_size))
        for n_filter in n_filters[1:]:
            model.add(Conv2D(n_filter, kernel_size=kernel_size,
                      strides=strides, padding=padding_method))
            model.add(Activation("relu"))
            if maxpool:
                model.add(MaxPooling1D(
                    pool_size=pool_size,
                    strides=maxpool_strides,
                    padding=maxpool_padding_method,  # Padding method
                ))
        model.add(Flatten())
        return model

    def create_dnn_layer(self, n_nodes=[128, 64, 32], name='output', l2_norm=0.01, use_spectral_norm=False, dropout=0.5):
        model = tf.keras.Sequential()
        if use_spectral_norm:
            for n_node in n_nodes:
                model.add(SpectralNormalization(Dense(
                    n_node, kernel_regularizer=l2(l2_norm), activation='relu'
                ), dynamic=True))
                model.add(BatchNormalization())
                model.add(Dropout(dropout))
            model.add(SpectralNormalization(Dense(1), dynamic=True))

        else:
            for n_node in n_nodes:
                model.add(Dense(n_node, kernel_regularizer=l2(
                    l2_norm), activation='relu'))
                model.add(BatchNormalization())
                model.add(Dropout(dropout))
            model.add(Dense(1))

        return model

    def create(self):
        lr = self.model_params['lr']
        clipnorm = self.model_params['clipnorm']
        pcc_rmse_alpha = self.model_params['pcc_rmse_alpha']

        radial_cnn = self.create_cnn2d_layer(
            **self.model_params['radial_cnn_params'])
        angular_cnn = self.create_cnn2d_layer(
            **self.model_params['angular_cnn_params'])
        dnn = self.create_dnn_layer(**self.model_params['dnn_params'])

        # Input(shape=(8, 8, 8))
        input_radial = Input(
            shape=self.model_params['radial_cnn_params']['input_size'])
        input_angular = Input(
            shape=self.model_params['angular_cnn_params']['input_size'])
        x_radial = radial_cnn(input_radial)
        x_angular = angular_cnn(input_angular)
        x = concatenate([x_radial, x_angular])  # x_radial + x_angular
        output = dnn(x)

        model = tf.keras.models.Model(
            inputs=[input_radial, input_angular],
            outputs=[output]  # [output, output_cmp, output_lgd]
        )

        # optimizer = SGD(
        #     lr=lr,
        #     momentum=0.9,
        #     decay=1e-6,
        #     # clipnorm=clipnorm
        # )

        # loss_pcc_rmse = make_PCC_RMSE_with_alpha(alpha=pcc_rmse_alpha)

        # model.compile(
        #     loss=loss_pcc_rmse,
        #     optimizer=optimizer,
        #     metrics=[RMSE, PCC, loss_pcc_rmse]
        # )
        model = self.compile(model)

        return model

    def compile(self, model):
        """function for compiling the network.

        Args:
            network ([type]): [description]

        Returns:
            [type]: [description]
        """
        clipnorm = self.model_params['clipnorm']
        pcc_rmse_alpha = self.model_params['pcc_rmse_alpha']
        lr = self.model_params['lr']

        optimizer = SGD(
            lr=lr,
            momentum=0.9,
            decay=1e-6,
            # clipnorm=clipnorm
        )

        loss_pcc_rmse = make_PCC_RMSE_with_alpha(alpha=pcc_rmse_alpha)

        model.compile(
            loss=loss_pcc_rmse,
            optimizer=optimizer,
            metrics=[RMSE, PCC, loss_pcc_rmse]
        )

        return model

    def _preprocess_pandas(self, X, y):
        pf = self.PreprocessFunction(
            target_elements=self.feature_params['target_elements'],
            n_radial_Rs=self.feature_params['n_radial_Rs'],
            n_angular_Rs=self.feature_params['n_angular_Rs'],
            n_thetas=self.feature_params['n_thetas'],
        )
        processed_feature, processed_label = pf.process_pandas(X, y)
        return processed_feature, processed_label

    def _preprocess_tfrecord(self, tfrecords):
        """Function to preprocess input of data type TFRecords.

        Args:
            tfrecords ([TFRecordDatasetV2]): Input and target data. 

        Returns:
            [TFRecordDatasetV2]: Pre-processed input and target data.
        """
        pf = self.PreprocessFunction(
            target_elements=self.feature_params['target_elements'],
            n_radial_Rs=self.feature_params['n_radial_Rs'],
            n_angular_Rs=self.feature_params['n_angular_Rs'],
            n_thetas=self.feature_params['n_thetas'],
        )
        return tfrecords.map(pf.parse_batched_example).map(pf.process_tfrecord)

    def _preprocess_batched_tfrecord(self, batched_tfrecord):
        """Function to preprocess input of data type TFRecords.

        Args:
            tfrecords ([TFRecordDatasetV2]): Input and target data. 

        Returns:
            [TFRecordDatasetV2]: Pre-processed input and target data.
        """
        pf = self.PreprocessFunction(
            target_elements=self.feature_params['target_elements'],
            n_radial_Rs=self.feature_params['n_radial_Rs'],
            n_angular_Rs=self.feature_params['n_angular_Rs'],
            n_thetas=self.feature_params['n_thetas'],
        )
        return batched_tfrecord.map(pf.parse_batched_example).map(pf.process_batched_tfrecord)

    def preprocess(self, X, y=None):
        """Pre-process input and target data into a form acceptable to the model.

        Args:
            X : Input data. Pandas DataFrame of TFRecordDatasetV2 can be accepted.
            y ([type], optional): Target data. If the type of X is TFRecordDatasetV2 then y arg is ignored. Defaults to None.

        Returns:
            [type]: Pre-processed input and target data
        """
        if isinstance(X, BatchDataset):
            return self._preprocess_batched_tfrecord(X)
        elif isinstance(X, TFRecordDataset):
            return self._preprocess_tfrecord(X)
        elif isinstance(X, pd.core.frame.DataFrame):
            return self._preprocess_pandas(X, y)
        else:
            TypeError('''Feature(X) and label(y) must be pandas.DataFrame or TFRecordDataset.
            Please check the type of your data.''')

    class PreprocessFunction(object):
        """Class for creating functions for preprocessing input and target data.
        Specializes in preprocessing CNN2D class input data.
        In the CNN2D class, the feature names are very important because the feature partitioning depends on the names
        This class determines the element of the ligand from the feature name using regular expressions and divide the feature by the element of the ligand.

        Args:
            object ([type]): [description]
        """

        def __init__(self,  # radial_feature_dim=512, angular_feature_dim=32768,
                     target_elements=['H', 'C', 'N',
                                      'O', 'P', 'S', 'Cl', 'DU'],
                     n_radial_Rs=8, n_angular_Rs=8, n_thetas=8,
                     ):
            """
            In the CNN2D class, the feature names are very important because the feature partitioning depends on the names.
            This class determines the element of the ligand from the feature name using regular expressions and divide the feature by the element of the ligand.
            Therefore, this class needs feature_names, target_elements and regex_template.

            Args:
                feature_names ([list, pd.core.strings.StringMethods]): feature names.
                target_elements ([list]): Element types to be considered. Default to ['H', 'C', 'N', 'O', 'P', 'S', 'Cl', 'DU'].
            """
            self.target_elements = target_elements
            self.n_target_elements = len(target_elements)
            self.n_radial_Rs = n_radial_Rs
            self.n_angular_Rs = n_angular_Rs
            self.n_thetas = n_thetas
            self.radial_feature_dim = self.n_target_elements**2 * \
                n_radial_Rs  # radial_feature_dim
            self.angular_feature_dim = self.n_target_elements**3 * \
                n_angular_Rs * n_thetas  # angular_feature_dim
            self.complex_feature_dim = self.radial_feature_dim + self.angular_feature_dim
            return None

        def process_pandas(self, feature, label=None):
            """Preprocess feature and label of data type pandas DataFrame.

            Args:
                feature ([pandas.core.frame.DataFrame]): Input data.
                label ([pandas.core.frame.DataFrame], optional): Target data. Defaults to None.

            Returns:
                processed_feature [pandas.core.frame.DataFrame]: Processed input data.
                processed_label [pandas.core.frame.DataFrame]: Processed target data.
            """
            radial_feature = feature.values[:, :self.radial_feature_dim]
            radial_feature = tf.reshape(radial_feature,
                                        (-1, self.n_target_elements,
                                         self.n_target_elements, self.n_radial_Rs)
                                        )
            radial_feature = tf.transpose(radial_feature, (0, 2, 3, 1))

            angular_feature = feature.values[:,
                                             self.radial_feature_dim: self.radial_feature_dim + self.angular_feature_dim]
            angular_feature = tf.reshape(angular_feature,
                                         (-1, self.n_target_elements, self.n_target_elements, self.n_target_elements, self.n_angular_Rs*self.n_thetas))
            angular_feature = tf.transpose(angular_feature, (0, 1, 3, 4, 2))
            angular_feature = tf.reshape(angular_feature,
                                         (-1, self.n_target_elements*self.n_target_elements,
                                          self.n_angular_Rs*self.n_thetas, self.n_target_elements)
                                         )

            processed_label = label.values[:, 0].astype(
                "float64").copy() if label is not None else None
            return (radial_feature, angular_feature), processed_label

        @tf.autograph.experimental.do_not_convert
        def parse_example(self, example):
            """parse tfrecord to feature and label.

            Args:
                example ([tensorflow.python.framework.ops.Tensor]): [description]

            Returns:
                x [tensorflow.python.framework.ops.Tensor]: [description]
                y [tensorflow.python.framework.ops.Tensor]: [description]
            """
            # feature_dim = 33280
            features = tf.io.parse_single_example(
                example,
                features={
                    'complex_feature': tf.io.FixedLenFeature([self.complex_feature_dim], dtype=tf.float32),
                    'label': tf.io.FixedLenFeature([1], dtype=tf.float32),
                    'sample_id': tf.io.FixedLenFeature([1], dtype=tf.string)
                }
            )
            complex_feature = features["complex_feature"]
            label = features["label"]
            sample_id = features["sample_id"]

            return complex_feature, label, sample_id

        @tf.autograph.experimental.do_not_convert
        def parse_batched_example(self, example):
            """parse tfrecord to feature and label.

            Args:
                example ([tensorflow.python.framework.ops.Tensor]): [description]

            Returns:
                x [tensorflow.python.framework.ops.Tensor]: [description]
                y [tensorflow.python.framework.ops.Tensor]: [description]
            """
            # feature_dim = 33280
            features = tf.io.parse_example(
                example,
                features={
                    'complex_feature': tf.io.FixedLenFeature([self.complex_feature_dim], dtype=tf.float32),
                    'label': tf.io.FixedLenFeature([1], dtype=tf.float32),
                    'sample_id': tf.io.FixedLenFeature([1], dtype=tf.string)
                }
            )
            complex_feature = features["complex_feature"]
            label = features["label"]
            sample_id = features["sample_id"]

            return complex_feature, label, sample_id

        @tf.autograph.experimental.do_not_convert
        def process_tfrecord(self, complex_feature, label, sample_id):
            """Preprocess feature and label parsed from example by parse_example function above.

            Args:
                feature ([type]): [description]
                label ([type]): [description]

            Returns:
                [type]: [description]
            """
            # add scaling phase?
            # radial_feature = complex_feature[:self.radial_feature_dim]
            # radial_feature = tf.reshape(radial_feature, (8, 8, 8))
            # radial_feature = tf.transpose(radial_feature, (1, 2, 0)) #tf.transpose(tf.reshape(radial_feature, (-1, 8, 8, 8)), (0, 3, 2, 1))

            # angular_feature = complex_feature[self.radial_feature_dim: self.radial_feature_dim + self.angular_feature_dim]
            # angular_feature = tf.reshape(angular_feature, (8, 8, 8, 64))
            # angular_feature = tf.transpose(angular_feature, [0, 2, 3, 1])
            # angular_feature = tf.reshape(angular_feature, (64, 64, 8))

            radial_feature = complex_feature[:self.radial_feature_dim]
            radial_feature = tf.reshape(radial_feature,
                                        (self.n_target_elements,
                                         self.n_target_elements, self.n_radial_Rs)
                                        )
            radial_feature = tf.transpose(radial_feature, (1, 2, 0))

            angular_feature = complex_feature[self.radial_feature_dim:
                                              self.radial_feature_dim + self.angular_feature_dim]
            angular_feature = tf.reshape(angular_feature,
                                         (self.n_target_elements, self.n_target_elements, self.n_target_elements, self.n_angular_Rs*self.n_thetas))
            angular_feature = tf.transpose(angular_feature, [0, 2, 3, 1])
            angular_feature = tf.reshape(angular_feature,
                                         (self.n_target_elements*self.n_target_elements,
                                          self.n_angular_Rs*self.n_thetas, self.n_target_elements)
                                         )

            return (radial_feature, angular_feature), label

        @tf.autograph.experimental.do_not_convert
        def process_batched_tfrecord(self, complex_feature, label, sample_id):
            """Preprocess feature and label parsed from example by parse_example function above.

            Args:
                feature ([type]): [description]
                label ([type]): [description]

            Returns:
                [type]: [description]
            """
            # add scaling phase?
            # radial_feature = complex_feature[:self.radial_feature_dim]
            # radial_feature = tf.reshape(radial_feature, (8, 8, 8))
            # radial_feature = tf.transpose(radial_feature, (1, 2, 0)) #tf.transpose(tf.reshape(radial_feature, (-1, 8, 8, 8)), (0, 3, 2, 1))

            # angular_feature = complex_feature[self.radial_feature_dim: self.radial_feature_dim + self.angular_feature_dim]
            # angular_feature = tf.reshape(angular_feature, (8, 8, 8, 64))
            # angular_feature = tf.transpose(angular_feature, [0, 2, 3, 1])
            # angular_feature = tf.reshape(angular_feature, (64, 64, 8))

            radial_feature = complex_feature[:, :self.radial_feature_dim]
            radial_feature = tf.reshape(radial_feature,
                                        (-1, self.n_target_elements,
                                         self.n_target_elements, self.n_radial_Rs)
                                        )
            radial_feature = tf.transpose(radial_feature, (0, 2, 3, 1))

            angular_feature = complex_feature[:,
                                              self.radial_feature_dim: self.radial_feature_dim + self.angular_feature_dim]
            angular_feature = tf.reshape(angular_feature,
                                         (-1, self.n_target_elements, self.n_target_elements, self.n_target_elements, self.n_angular_Rs*self.n_thetas))
            angular_feature = tf.transpose(angular_feature, (0, 1, 3, 4, 2))
            angular_feature = tf.reshape(angular_feature,
                                         (-1, self.n_target_elements*self.n_target_elements,
                                          self.n_angular_Rs*self.n_thetas, self.n_target_elements)
                                         )

            return (radial_feature, angular_feature), label

    # def generate_label(self, tfrecords, batch_size=128):
    #     labels_dataset = self.preprocess(tfrecords).map(lambda feature, labels, sample_id: labels).batch(batch_size)
    #     labels_array = np.concatenate([
    #         np.hstack([col.numpy().reshape([-1,1]) for col in label]) for label in labels_dataset])
    #     labels_index = self.generate_index(tfrecords)
    #     labels = self.shape_preds(labels_array, index=labels_index)
    #     return labels

    @tf.autograph.experimental.do_not_convert
    def generate_label(self, tfrecords, batch_size=128):
        pf = self.PreprocessFunction(
            target_elements=self.feature_params['target_elements'],
            n_radial_Rs=self.feature_params['n_radial_Rs'],
            n_angular_Rs=self.feature_params['n_angular_Rs'],
            n_thetas=self.feature_params['n_thetas'],
        )
        def get_labels(feature, labels, sample_id): return labels
        labels_dataset = tfrecords.batch(batch_size)\
            .map(pf.parse_batched_example)\
            .map(get_labels)
        labels_array = np.concatenate([
            np.hstack([col.numpy().reshape([-1, 1]) for col in label])
            for label in labels_dataset
        ])
        labels_index = self.generate_index(tfrecords)
        labels = self.shape_preds(labels_array, index=labels_index)
        return labels

    @tf.autograph.experimental.do_not_convert
    def generate_index(self, tfrecords, batch_size=128):
        pf = self.PreprocessFunction(
            target_elements=self.feature_params['target_elements'],
            n_radial_Rs=self.feature_params['n_radial_Rs'],
            n_angular_Rs=self.feature_params['n_angular_Rs'],
            n_thetas=self.feature_params['n_thetas']
        )
        def get_sample_id(feature, label, sample_id): return sample_id
        index_dataset = tfrecords.batch(batch_size)\
            .map(pf.parse_batched_example)\
            .map(get_sample_id)
        index = np.vstack([idx for idx in index_dataset]).flatten()
        index = np.vectorize(lambda bytes_obj: bytes_obj.decode())(index)
        return index

    def shape_preds(self, preds, index):
        """function for formatting prediction results into pandas.DataFrame.

        Args:
            preds ([type]): Values of the result dataframe.
            index ([type]): Index of the result dataframe.

        Returns:
            [type]: Formatted prediction result.
        """
        if isinstance(preds, list):
            shaped_preds = pd.DataFrame(np.concatenate(
                preds, axis=1), columns=self.label_colnames, index=index)
        else:
            shaped_preds = pd.DataFrame(
                preds, columns=self.label_colnames, index=index)
        return shaped_preds

    # TFRecord file generation related methods
    @staticmethod
    def _float_feature(value):
        """return float_list from float / double """
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    @staticmethod
    def _bytes_feature(value):
        """return byte_list from string / byte """
        if isinstance(value, type(tf.constant(0))):
            # BytesList won't unpack a string from an EagerTensor.
            value = value.numpy()
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    @staticmethod
    def record2example(cmp_feature, label, sample_id):
        return tf.train.Example(features=tf.train.Features(feature={
            'complex_feature': CNN2D._float_feature(cmp_feature),
            'label': CNN2D._float_feature(label),
            'sample_id': CNN2D._bytes_feature(sample_id)
        }))

    @staticmethod
    def _write_feature_to_tfrecords(complex_feature_files, label_file,
                                    tfr_filename, label_colnames=[
                                        'pKa_energy'],
                                    scheduler='processes', feature_index_replace=None):  # feature_index_replace='_complex
        # load features and labels
        feature_cmp = runner.load_feature(
            complex_feature_files, scheduler=scheduler).astype('float32')
        label = runner.load_label(
            label_file, label_colnames=label_colnames).astype('float32')

        # process feature_cmp index
        if feature_index_replace is not None:
            feature_cmp.index = [idx.replace(
                feature_index_replace, '') for idx in feature_cmp.index]

        # remove not common index
        common_idx = feature_cmp.index.intersection(label.index)
        feature_cmp = feature_cmp.loc[common_idx]
        label = label.loc[common_idx]

        # check feature
        if not all(feature_cmp.index == label.index):
            raise ValueError('feature_cmp and label have different indexes.')
        if len(feature_cmp.index) == 0:
            raise ValueError(
                'There are no features to write. Length of feature is 0.')

        # create byte type sample_id
        sample_id = np.vectorize(lambda x: x.encode())(label.index.values)

        # write data to tfr_filename
        n_sample = sample_id.shape[0]
        with tf.io.TFRecordWriter(tfr_filename) as writer:
            for i in tqdm(range(n_sample)):
                ex = CNN2D.record2example(
                    cmp_feature=feature_cmp.values[i],
                    label=label.values[i],
                    sample_id=sample_id[i]
                )
                writer.write(ex.SerializeToString())
        return None

    @staticmethod
    def generate_feature_tfrecords(complex_feature_files,
                                   label_file,
                                   tfr_filename,
                                   label_colnames=['pKa_energy'],
                                   scheduler='processes'):
        CNN2D._write_feature_to_tfrecords(
            complex_feature_files,
            label_file,
            tfr_filename,
            label_colnames=label_colnames,
            scheduler=scheduler
        )
        print(f"{tfr_filename} was saved...")
        return None


class CNN1D(MyAbstractNetworkStructure):
    def __init__(self, input_size=(33280, 1),  label_colnames=['pKa'],
                 feature_dim=33280, target_elements=['H', 'C', 'N', 'O', 'P', 'S', 'Cl', 'DU'], n_radial_Rs=8, n_angular_Rs=8, n_thetas=8,
                 lr=0.0001, clipnorm=1, pcc_rmse_alpha=0.7,
                 n_filters=[32, 64, 128], kernel_size=4, strides=1, padding_method='valid', maxpool=False, pool_size=2, maxpool_strides=2, maxpool_padding_method='same',
                 n_nodes=[128, 64, 32], l2_norm=0.01, use_spectral_norm=False, dropout=0.5
                 ):
        """[summary]

        Args:
            input_size (tuple, optional): Model Parameter. [description]. Defaults to (33280,1).
            label_colnames (list, optional):  Column names of the target label. Defaults to ['pKa'].
            feature_dim (int, optional): Feature Prameter. Number of the dimension of the feature. Defaults to 33280.
            target_elements (list, optional): Feature Prameter. Element types considered in feature generation step. Defaults to ['H', 'C', 'N', 'O', 'P', 'S', 'Cl', 'DU'].
            n_radial_Rs (int, optional): Feature Prameter. Number of Rs args in radial symmetry function. Defaults to 8.
            n_angular_Rs (int, optional): Feature Prameter. Number of Rs args in angular symmetry function. Defaults to 8.
            n_thetas (int, optional): Feature Prameter. Number of theta args in angular symmetry function.  Defaults to 8.
            lr (float, optional): Model Parameter. [description]. Defaults to 0.0001.
            clipnorm (int, optional): Model Parameter. [description]. Defaults to 1.
            pcc_rmse_alpha (float, optional): Model Parameter. Ratio of PCC to RMSE in the loss function. Defaults to 0.7.
            n_filters (list, optional): Model Parameter (CNN). List of the number of the filters per layer. Defaults to [32, 64, 128].
            kernel_size (int, optional):  Model Parameter (CNN). [description]. Defaults to 4.
            strides (int, optional):  Model Parameter (CNN). [description]. Defaults to 1.
            padding_method (str, optional):  Model Parameter (CNN). [description]. Defaults to 'valid'.
            maxpool (bool, optional):  Model Parameter (CNN). description]. Defaults to False.
            pool_size (int, optional):  Model Parameter (CNN). [description]. Defaults to 2.
            maxpool_strides (int, optional):  Model Parameter (CNN). [description]. Defaults to 2.
            maxpool_padding_method (str, optional):  Model Parameter (CNN). [description]. Defaults to 'same'.
            n_nodes (list, optional):  Model Parameter (DNN). List of the number of the nodes per layer. Defaults to [128, 64, 32].
            l2_norm (float, optional): Model Parameter (DNN). [description]. Defaults to 0.01.
            use_spectral_norm (bool, optional): Model Parameter (DNN). [description]. Defaults to False.
            dropout (float, optional): Model Parameter (DNN). [description]. Defaults to 0.5.

        Returns:
            [type]: [description]
        """

        self.label_colnames = label_colnames
        self.model_params = dict(
            input_size=input_size,  # 32768/8 + 512/8  # prev ver: 576, # 4096/8 + 512/8,
            lr=lr,
            clipnorm=clipnorm,
            pcc_rmse_alpha=pcc_rmse_alpha,
            cnn_params=dict(
                n_filters=n_filters,
                kernel_size=kernel_size,
                strides=strides,
                padding_method=padding_method,
                maxpool=maxpool,
                pool_size=pool_size,
                maxpool_strides=maxpool_strides,
                maxpool_padding_method=maxpool_padding_method,
            ),
            dnn_params=dict(
                n_nodes=n_nodes,
                l2_norm=l2_norm,
                use_spectral_norm=use_spectral_norm,
                dropout=dropout
            )
        )
        self.feature_params = dict(
            feature_dim=feature_dim,
            target_elements=target_elements,
            n_target_elements=len(target_elements),
            n_radial_Rs=n_radial_Rs,
            n_angular_Rs=n_angular_Rs,
            n_thetas=n_thetas
        )
        return None

    # , target_elements, n_radial_Rs, n_angular_Rs, n_thetas):
    def create_feature_name(self):
        """Create feature names from given parameters. Feature names are required for preprocessing.

        Args:
            self ([type]): [description]

        Returns:
            feature name [list]: List of radial feature name and angular feature names
        """
        target_elements = self.feature_params['target_elements']
        n_radial_Rs = self.feature_params['n_radial_Rs']
        n_angular_Rs = self.feature_params['n_angular_Rs']
        n_thetas = self.feature_params['n_thetas']

        radial_feature_name = [f"{e_l}_{e_r}_{rs}"
                               for e_l, e_r in itertools.product(target_elements, repeat=2)
                               for rs in range(n_radial_Rs)
                               ]
        angular_feature_name = [f"{e_j}_{e_i}_{e_k}_{theta}_{rs}"
                                for e_j, e_i, e_k in itertools.product(target_elements, repeat=3)
                                for theta in range(n_thetas)
                                for rs in range(n_angular_Rs)
                                ]
        return radial_feature_name + angular_feature_name

    def create_cnn_layer(self, input_size=(33280, 1), n_filters=[32, 64, 128], kernel_size=4, strides=1, padding_method='valid',
                         maxpool=False, pool_size=2, maxpool_strides=2, maxpool_padding_method='same'):
        model = tf.keras.Sequential()
        model.add(Conv1D(n_filters[0], kernel_size=kernel_size,
                  strides=strides, padding=padding_method, input_shape=input_size))
        for n_filter in n_filters[1:]:
            model.add(Conv1D(n_filter, kernel_size=kernel_size,
                      strides=strides, padding=padding_method))
            model.add(Activation("relu"))
            if maxpool:
                model.add(MaxPooling1D(
                    pool_size=pool_size,
                    strides=maxpool_strides,
                    padding=maxpool_padding_method,  # Padding method
                ))
        model.add(Flatten())
        return model

    def create_dnn_layer(self, n_nodes=[128, 64, 32], l2_norm=0.01, use_spectral_norm=False, dropout=0.5, name='output'):
        model = tf.keras.Sequential()
        if use_spectral_norm:
            for n_node in n_nodes:
                model.add(SpectralNormalization(Dense(
                    n_node, kernel_regularizer=l2(l2_norm), activation='relu'
                ), dynamic=True))
                model.add(BatchNormalization())
                model.add(Dropout(dropout))
            model.add(SpectralNormalization(Dense(1), dynamic=True))
        else:
            for n_node in n_nodes:
                model.add(Dense(n_node, kernel_regularizer=l2(
                    l2_norm), activation='relu'))
                model.add(BatchNormalization())
                model.add(Dropout(dropout))
            model.add(Dense(1, name=name))
        return model

    def create(self):
        input_size = self.model_params['input_size']
        lr = self.model_params['lr']
        clipnorm = self.model_params['clipnorm']
        pcc_rmse_alpha = self.model_params['pcc_rmse_alpha']
        cnn_params = self.model_params['cnn_params']
        dnn_params = self.model_params['dnn_params']

        input = Input(shape=input_size)
        cnn = self.create_cnn_layer(**cnn_params)
        dnn = self.create_dnn_layer(name='output', **dnn_params)

        x = cnn(input)
        output = dnn(x)

        model = tf.keras.models.Model(
            inputs=[input],
            outputs=[output],
            name='output'
        )

        optimizer = SGD(
            lr=lr,
            momentum=0.9,
            decay=1e-6,
            # clipnorm=clipnorm
        )

        loss_pcc_rmse = make_PCC_RMSE_with_alpha(alpha=pcc_rmse_alpha)

        # model.compile(
        #     loss={'output': loss_pcc_rmse},
        #     optimizer=optimizer,
        #     metrics={'output': ['mse', RMSE, PCC, loss_pcc_rmse]}
        # )
        model.compile(
            loss=loss_pcc_rmse,
            optimizer=optimizer,
            metrics=['mse', RMSE, PCC, loss_pcc_rmse]
        )
        return model

    def _preprocess_tfrecord(self, tfrecords):
        """Function to preprocess input of data type TFRecords.

        Args:
            tfrecords ([TFRecordDatasetV2]): Input and target data. 

        Returns:
            [TFRecordDatasetV2]: Pre-processed input and target data.
        """
        # feature_names = self.create_feature_name()
        pf = self.PreprocessFunction(
            # complex_feature_names=feature_names,
            complex_feature_dim=self.feature_params['feature_dim'],
            target_elements=self.feature_params['target_elements']
        )
        return tfrecords.map(pf.parse_example).map(pf.process_tfrecord)

    def preprocess(self, X, y=None):
        """Pre-process input and target data into a form acceptable to the model.

        Args:
            X : Input data. Pandas DataFrame of TFRecordDatasetV2 can be accepted.
            y ([type], optional): Target data. If the type of X is TFRecordDatasetV2 then y arg is ignored. Defaults to None.

        Returns:
            [type]: Pre-processed input and target data
        """
        if isinstance(X, TFRecordDataset):
            return self._preprocess_tfrecord(X)
        elif isinstance(X, pd.core.frame.DataFrame):
            return self._preprocess_pandas(X, y)
        else:
            TypeError('''Feature(X) and label(y) must be pandas.DataFrame or TFRecordDataset.
            Please check the type of your data.''')

    class PreprocessFunction(object):
        """Class for creating functions for preprocessing input and target data.
        Specializes in preprocessing CNN1D class input data.
        In the CNN1D class, the feature names are very important because the feature partitioning depends on the names
        This class determines the element of the ligand from the feature name using regular expressions and divide the feature by the element of the ligand.

        Args:
            object ([type]): [description]
        """

        def __init__(self, complex_feature_dim=33280, complex_feature_names=None, target_elements=['H', 'C', 'N', 'O', 'P', 'S', 'Cl', 'DU']):
            """instantiate by passing feature name. 
            In the CNN1D class, the feature names are very important because the feature partitioning depends on the names.
            This class determines the element of the ligand from the feature name using regular expressions and divide the feature by the element of the ligand.
            Therefore, this class needs feature_names, target_elements and regex_template.

            Args:
                feature_names ([list, pd.core.strings.StringMethods]): feature names.
                target_elements ([list]): Element types to be considered. Default to ['H', 'C', 'N', 'O', 'P', 'S', 'Cl', 'DU'].
            """
            # self.complex_feature_names = pd.Index(complex_feature_names).str \
            #     if not isinstance(complex_feature_names, pd.core.strings.StringMethods) else complex_feature_names
            # self.complex_feature_names.__dict__['_orig'].size
            self.complex_feature_dim = complex_feature_dim
            # self.target_elements = target_elements #['H', 'C', 'N', 'O', 'P', 'S', 'Cl', 'DU']
            # self.regex_template = "(^\w?\w?_{element}_\w?\w?_\d+_\d+$)|(^{element}_\w?\w?_\d+$)"
            return None

        def process_pandas(self, feature, label=None):
            """Preprocess feature and label of data type pandas DataFrame.

            Args:
                feature ([pandas.core.frame.DataFrame]): Input data.
                label ([pandas.core.frame.DataFrame], optional): Target data. Defaults to None.

            Returns:
                processed_feature [pandas.core.frame.DataFrame]: Processed input data.
                processed_label [pandas.core.frame.DataFrame]: Processed target data.
            """
            processed_feature = feature
            processed_label = label.values[:, 0].astype(
                "float64").copy() if label is not None else None
            return processed_feature, processed_label

        # @staticmethod
        def parse_example(self, example):
            """parse tfrecord to feature and label.

            Args:
                example ([tensorflow.python.framework.ops.Tensor]): [description]

            Returns:
                x [tensorflow.python.framework.ops.Tensor]: [description]
                y [tensorflow.python.framework.ops.Tensor]: [description]
            """
            # feature_dim = 33280
            features = tf.io.parse_single_example(
                example,
                features={
                    'complex_feature': tf.io.FixedLenFeature([self.complex_feature_dim], dtype=tf.float32),
                    'label': tf.io.FixedLenFeature([1], dtype=tf.float32),
                    'sample_id': tf.io.FixedLenFeature([1], dtype=tf.string)
                }
            )
            complex_feature = features["complex_feature"]
            label = features["label"]
            sample_id = features["sample_id"]

            return complex_feature, label, sample_id

        def process_tfrecord(self, complex_feature, label, sample_id):
            """Preprocess feature and label parsed from example by parse_example function above.

            Args:
                feature ([type]): [description]
                label ([type]): [description]

            Returns:
                [type]: [description]
            """
            # add scaling phase?
            processed_complex_feature = complex_feature

            return processed_complex_feature, label

    def generate_label(self, tfrecords, batch_size=128):
        labels_dataset = self.preprocess(tfrecords).map(
            lambda feature, labels: labels).batch(batch_size)
        labels_array = np.concatenate([np.hstack(
            [col.numpy().reshape([-1, 1]) for col in label]) for label in labels_dataset])
        labels_index = self.generate_index(tfrecords)
        labels = self.shape_preds(labels_array, index=labels_index)
        return labels

    def generate_index(self, tfrecords, batch_size=128):
        feature_names = self.create_feature_name()
        pf = self.PreprocessFunction(
            complex_feature_names=feature_names,
            target_elements=self.feature_params['target_elements']
        )
        index_dataset = tfrecords.map(pf.parse_example).map(
            lambda complex_feature, label, sample_id: sample_id).batch(batch_size)
        index = np.vstack([idx for idx in index_dataset]).flatten()
        index = np.vectorize(lambda bytes_obj: bytes_obj.decode())(index)
        return index

    def shape_preds(self, preds, index):
        """function for formatting prediction results into pandas.DataFrame.

        Args:
            preds ([type]): Values of the result dataframe.
            index ([type]): Index of the result dataframe.

        Returns:
            [type]: Formatted prediction result.
        """
        if isinstance(preds, list):
            shaped_preds = pd.DataFrame(np.concatenate(
                preds, axis=1), columns=self.label_colnames, index=index)
        else:
            shaped_preds = pd.DataFrame(
                preds, columns=self.label_colnames, index=index)
        return shaped_preds

    # TFRecord file generation related methods
    @staticmethod
    def _float_feature(value):
        """return float_list from float / double """
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    @staticmethod
    def _bytes_feature(value):
        """return byte_list from string / byte """
        if isinstance(value, type(tf.constant(0))):
            # BytesList won't unpack a string from an EagerTensor.
            value = value.numpy()
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    @staticmethod
    def record2example(cmp_feature, label, sample_id):
        return tf.train.Example(features=tf.train.Features(feature={
            'complex_feature': CNN1D._float_feature(cmp_feature),
            'label': CNN1D._float_feature(label),
            'sample_id': CNN1D._bytes_feature(sample_id)
        }))

    @staticmethod
    def _write_feature_to_tfrecords(complex_feature_files, label_file,
                                    tfr_filename, label_colnames=['pKa']):
        # load features and labels
        feature_cmp = runner.load_feature(
            complex_feature_files).astype('float32')
        label = runner.load_label(
            label_file, label_colnames=label_colnames).astype('float32')

        # process feature_cmp index
        feature_cmp.index = [idx.replace('_complex', '')
                             for idx in feature_cmp.index]

        # remove not common index
        common_idx = feature_cmp.index.intersection(label.index)
        feature_cmp = feature_cmp.loc[common_idx]
        label = label.loc[common_idx]

        # check feature
        if not all(feature_cmp.index == label.index):
            raise ValueError('feature_cmp and label have different indexes.')
        if len(feature_cmp.index) == 0:
            raise ValueError(
                'There are no features to write. Length of feature is 0.')

        # create byte type sample_id
        sample_id = np.vectorize(lambda x: x.encode())(label.index.values)

        # write data to tfr_filename
        n_sample = sample_id.shape[0]
        with tf.io.TFRecordWriter(tfr_filename) as writer:
            for i in tqdm(range(n_sample)):
                ex = CNN1D.record2example(
                    cmp_feature=feature_cmp.values[i],
                    label=label.values[i],
                    sample_id=sample_id[i]
                )
                writer.write(ex.SerializeToString())
        return None

    @staticmethod
    def generate_feature_tfrecords(complex_feature_files, label_file, tfr_filename, label_colnames=['pKa']):
        CNN1D._write_feature_to_tfrecords(
            complex_feature_files,
            label_file,
            tfr_filename,
            label_colnames=label_colnames
        )
        print(f"{tfr_filename} was saved...")
        return None


class RadAngDNN(MyAbstractNetworkStructure):
    def __init__(self, label_colnames=['pKa'],
                 target_elements=['H', 'C', 'N', 'O', 'P', 'S', 'Cl', 'DU'], n_radial_Rs=8, n_angular_Rs=8, n_thetas=8,
                 lr=0.0001, clipnorm=1, pcc_rmse_alpha=0.7,
                 radial_dnn_n_nodes=[400, 200, 100], angular_dnn_n_nodes=[400, 200, 100], joined_dnn_n_nodes=[200, 100, 50],
                 l2_norm=0.01, use_spectral_norm=False, dropout=0.5
                 ):
        """[summary]

        Args:
            input_size (tuple, optional): Model Parameter. [description]. Defaults to (33280,1).
            label_colnames (list, optional):  Column names of the target label. Defaults to ['pKa'].
            feature_dim (int, optional): Feature Prameter. Number of the dimension of the feature. Defaults to 33280.
            target_elements (list, optional): Feature Prameter. Element types considered in feature generation step. Defaults to ['H', 'C', 'N', 'O', 'P', 'S', 'Cl', 'DU'].
            n_radial_Rs (int, optional): Feature Prameter. Number of Rs args in radial symmetry function. Defaults to 8.
            n_angular_Rs (int, optional): Feature Prameter. Number of Rs args in angular symmetry function. Defaults to 8.
            n_thetas (int, optional): Feature Prameter. Number of theta args in angular symmetry function.  Defaults to 8.
            lr (float, optional): Model Parameter. [description]. Defaults to 0.0001.
            clipnorm (int, optional): Model Parameter. [description]. Defaults to 1.
            pcc_rmse_alpha (float, optional): Model Parameter. Ratio of PCC to RMSE in the loss function. Defaults to 0.7.
            n_filters (list, optional): Model Parameter (CNN). List of the number of the filters per layer. Defaults to [32, 64, 128].
            kernel_size (int, optional):  Model Parameter (CNN). [description]. Defaults to 4.
            strides (int, optional):  Model Parameter (CNN). [description]. Defaults to 1.
            padding_method (str, optional):  Model Parameter (CNN). [description]. Defaults to 'valid'.
            maxpool (bool, optional):  Model Parameter (CNN). description]. Defaults to False.
            pool_size (int, optional):  Model Parameter (CNN). [description]. Defaults to 2.
            maxpool_strides (int, optional):  Model Parameter (CNN). [description]. Defaults to 2.
            maxpool_padding_method (str, optional):  Model Parameter (CNN). [description]. Defaults to 'same'.
            n_nodes (list, optional):  Model Parameter (DNN). List of the number of the nodes per layer. Defaults to [128, 64, 32].
            l2_norm (float, optional): Model Parameter (DNN). [description]. Defaults to 0.01.
            use_spectral_norm (bool, optional): Model Parameter (DNN). [description]. Defaults to False.
            dropout (float, optional): Model Parameter (DNN). [description]. Defaults to 0.5.

        Returns:
            [type]: [description]
        """
        n_target_elements = len(target_elements)
        radial_feature_dim = n_radial_Rs*(n_target_elements**2)
        angular_feature_dim = n_angular_Rs*n_thetas*(n_target_elements**3)

        self.label_colnames = label_colnames
        self.model_params = dict(
            # input_size=input_size, # 32768/8 + 512/8  # prev ver: 576, # 4096/8 + 512/8,
            lr=lr,
            clipnorm=clipnorm,
            pcc_rmse_alpha=pcc_rmse_alpha,
            radial_dnn_params=dict(
                input_size=radial_feature_dim,
                n_nodes=radial_dnn_n_nodes,
                l2_norm=l2_norm,
                use_spectral_norm=use_spectral_norm,
                dropout=dropout
            ),
            angular_dnn_params=dict(
                input_size=angular_feature_dim,
                n_nodes=angular_dnn_n_nodes,
                l2_norm=l2_norm,
                use_spectral_norm=use_spectral_norm,
                dropout=dropout
            ),
            joined_dnn_params=dict(
                n_nodes=joined_dnn_n_nodes,
                l2_norm=l2_norm,
                use_spectral_norm=use_spectral_norm,
                dropout=dropout
            )
        )
        self.feature_params = dict(
            target_elements=target_elements,
            n_target_elements=n_target_elements,
            n_radial_Rs=n_radial_Rs,
            n_angular_Rs=n_angular_Rs,
            n_thetas=n_thetas,
            radial_feature_dim=radial_feature_dim,
            angular_feature_dim=angular_feature_dim,
            feature_dim=radial_feature_dim + angular_feature_dim,
        )
        return None

    # , target_elements, n_radial_Rs, n_angular_Rs, n_thetas):
    def create_feature_name(self):
        """Create feature names from given parameters. Feature names are required for preprocessing.

        Args:
            self ([type]): [description]

        Returns:
            feature name [list]: List of radial feature name and angular feature names
        """
        target_elements = self.feature_params['target_elements']
        n_radial_Rs = self.feature_params['n_radial_Rs']
        n_angular_Rs = self.feature_params['n_angular_Rs']
        n_thetas = self.feature_params['n_thetas']

        radial_feature_name = [f"{e_l}_{e_r}_{rs}"
                               for e_l, e_r in itertools.product(target_elements, repeat=2)
                               for rs in range(n_radial_Rs)
                               ]
        angular_feature_name = [f"{e_j}_{e_i}_{e_k}_{theta}_{rs}"
                                for e_j, e_i, e_k in itertools.product(target_elements, repeat=3)
                                for theta in range(n_thetas)
                                for rs in range(n_angular_Rs)
                                ]
        return radial_feature_name + angular_feature_name

    def create_input_dnn_layer(self, input_size=None, n_nodes=[400, 200, 100], name='radial_dnn', l2_norm=0.01, use_spectral_norm=False, dropout=0.5):
        model = tf.keras.Sequential(name=name)
        if use_spectral_norm:
            for n_node in n_nodes:
                model.add(SpectralNormalization(Dense(
                    n_node, kernel_regularizer=l2(l2_norm), activation='relu'
                ), dynamic=True))
                model.add(BatchNormalization())
                model.add(Dropout(dropout))

        else:
            for n_node in n_nodes:
                model.add(Dense(n_node, kernel_regularizer=l2(
                    l2_norm), activation='relu'))
                model.add(BatchNormalization())
                model.add(Dropout(dropout))

        return model

    def create_output_dnn_layer(self, input_size=None, n_nodes=[400, 200, 100], name='output', l2_norm=0.01, use_spectral_norm=False, dropout=0.5):
        model = tf.keras.Sequential(name=name)
        if use_spectral_norm:
            for n_node in n_nodes:
                model.add(SpectralNormalization(Dense(
                    n_node, kernel_regularizer=l2(l2_norm), activation='relu'
                ), dynamic=True))
                model.add(BatchNormalization())
                model.add(Dropout(dropout))
            model.add(SpectralNormalization(Dense(1), dynamic=True))

        else:
            for n_node in n_nodes:
                model.add(Dense(n_node, kernel_regularizer=l2(
                    l2_norm), activation='relu'))
                model.add(BatchNormalization())
                model.add(Dropout(dropout))
            model.add(Dense(1))

        return model

    def create(self):
        lr = self.model_params['lr']
        clipnorm = self.model_params['clipnorm']
        pcc_rmse_alpha = self.model_params['pcc_rmse_alpha']

        radial_dnn = self.create_input_dnn_layer(
            **self.model_params['radial_dnn_params'], name='radial_dnn')
        angular_dnn = self.create_input_dnn_layer(
            **self.model_params['angular_dnn_params'], name='angular_dnn')
        joined_dnn = self.create_output_dnn_layer(
            **self.model_params['joined_dnn_params'], name='joined_dnn')

        # Input(shape=(8, 8, 8))
        input_radial = Input(
            shape=self.model_params['radial_dnn_params']['input_size'])
        input_angular = Input(
            shape=self.model_params['angular_dnn_params']['input_size'])
        x_radial = radial_dnn(input_radial)
        x_angular = angular_dnn(input_angular)
        x_joined = concatenate([x_radial, x_angular])  # x_radial + x_angular
        output = joined_dnn(x_joined)

        model = tf.keras.models.Model(
            inputs=[input_radial, input_angular],
            outputs=[output]  # [output, output_cmp, output_lgd]
        )

        optimizer = SGD(
            lr=lr,
            momentum=0.9,
            decay=1e-6,
            # clipnorm=clipnorm
        )

        loss_pcc_rmse = make_PCC_RMSE_with_alpha(alpha=pcc_rmse_alpha)

        model.compile(
            loss=loss_pcc_rmse,
            optimizer=optimizer,
            metrics=[RMSE, PCC, loss_pcc_rmse]
        )

        return model

    def _preprocess_pandas(self, X, y):
        pf = self.PreprocessFunction(
            target_elements=self.feature_params['target_elements'],
            n_radial_Rs=self.feature_params['n_radial_Rs'],
            n_angular_Rs=self.feature_params['n_angular_Rs'],
            n_thetas=self.feature_params['n_thetas'],
        )
        processed_feature, processed_label = pf.process_pandas(X, y)
        return processed_feature, processed_label

    def _preprocess_tfrecord(self, tfrecords):
        """Function to preprocess input of data type TFRecords.

        Args:
            tfrecords ([TFRecordDatasetV2]): Input and target data. 

        Returns:
            [TFRecordDatasetV2]: Pre-processed input and target data.
        """
        # feature_names = self.create_feature_name()
        pf = self.PreprocessFunction(
            # radial_feature_dim=self.feature_params['radial_feature_dim'],
            # angular_feature_dim=self.feature_params['angular_feature_dim'],
            target_elements=self.feature_params['target_elements'],
            n_radial_Rs=self.feature_params['n_radial_Rs'],
            n_angular_Rs=self.feature_params['n_angular_Rs'],
            n_thetas=self.feature_params['n_thetas']
        )
        return tfrecords.map(pf.parse_example).map(pf.process_tfrecord)

    def preprocess(self, X, y=None):
        """Pre-process input and target data into a form acceptable to the model.

        Args:
            X : Input data. Pandas DataFrame of TFRecordDatasetV2 can be accepted.
            y ([type], optional): Target data. If the type of X is TFRecordDatasetV2 then y arg is ignored. Defaults to None.

        Returns:
            [type]: Pre-processed input and target data
        """
        if isinstance(X, TFRecordDataset):
            return self._preprocess_tfrecord(X)
        elif isinstance(X, pd.core.frame.DataFrame):
            return self._preprocess_pandas(X, y)
        else:
            TypeError('''Feature(X) and label(y) must be pandas.DataFrame or TFRecordDataset.
            Please check the type of your data.''')

    class PreprocessFunction(object):
        """Class for creating functions for preprocessing input and target data.
        Specializes in preprocessing RadAngDNN class input data.
        In the RadAngDNN class, the feature names are very important because the feature partitioning depends on the names
        This class determines the element of the ligand from the feature name using regular expressions and divide the feature by the element of the ligand.

        Args:
            object ([type]): [description]
        """

        def __init__(self,  # radial_feature_dim=512, angular_feature_dim=32768,
                     target_elements=['H', 'C', 'N',
                                      'O', 'P', 'S', 'Cl', 'DU'],
                     n_radial_Rs=8, n_angular_Rs=8, n_thetas=8,
                     ):
            """
            In the RadAngDNN class, the feature names are very important because the feature partitioning depends on the names.
            This class determines the element of the ligand from the feature name using regular expressions and divide the feature by the element of the ligand.
            Therefore, this class needs feature_names, target_elements and regex_template.

            Args:
                feature_names ([list, pd.core.strings.StringMethods]): feature names.
                target_elements ([list]): Element types to be considered. Default to ['H', 'C', 'N', 'O', 'P', 'S', 'Cl', 'DU'].
            """
            # self.complex_feature_names = pd.Index(complex_feature_names).str \
            #     if not isinstance(complex_feature_names, pd.core.strings.StringMethods) else complex_feature_names
            self.target_elements = target_elements
            self.n_target_elements = len(target_elements)
            self.n_radial_Rs = n_radial_Rs
            self.n_angular_Rs = n_angular_Rs
            self.n_thetas = n_thetas
            self.radial_feature_dim = self.n_target_elements**2 * \
                n_radial_Rs  # radial_feature_dim
            self.angular_feature_dim = self.n_target_elements**3 * \
                n_angular_Rs * n_thetas  # angular_feature_dim
            self.complex_feature_dim = self.radial_feature_dim + self.angular_feature_dim
            # self.target_elements = target_elements #['H', 'C', 'N', 'O', 'P', 'S', 'Cl', 'DU']
            # self.regex_template = "(^\w?\w?_{element}_\w?\w?_\d+_\d+$)|(^{element}_\w?\w?_\d+$)"
            return None

        def process_pandas(self, feature, label=None):
            """Preprocess feature and label of data type pandas DataFrame.

            Args:
                feature ([pandas.core.frame.DataFrame]): Input data.
                label ([pandas.core.frame.DataFrame], optional): Target data. Defaults to None.

            Returns:
                processed_feature [pandas.core.frame.DataFrame]: Processed input data.
                processed_label [pandas.core.frame.DataFrame]: Processed target data.
            """

            radial_feature = feature.values[:, :self.radial_feature_dim]
            angular_feature = feature.values[:,
                                             self.radial_feature_dim: self.radial_feature_dim + self.angular_feature_dim]

            processed_label = label.values[:, 0].astype(
                "float64").copy() if label is not None else None
            return (radial_feature, angular_feature), processed_label

        # @staticmethod
        def parse_example(self, example):
            """parse tfrecord to feature and label.

            Args:
                example ([tensorflow.python.framework.ops.Tensor]): [description]

            Returns:
                x [tensorflow.python.framework.ops.Tensor]: [description]
                y [tensorflow.python.framework.ops.Tensor]: [description]
            """
            # feature_dim = 33280
            features = tf.io.parse_single_example(
                example,
                features={
                    'complex_feature': tf.io.FixedLenFeature([self.complex_feature_dim], dtype=tf.float32),
                    'label': tf.io.FixedLenFeature([1], dtype=tf.float32),
                    'sample_id': tf.io.FixedLenFeature([1], dtype=tf.string)
                }
            )
            complex_feature = features["complex_feature"]
            label = features["label"]
            sample_id = features["sample_id"]

            return complex_feature, label, sample_id

        def process_tfrecord(self, complex_feature, label, sample_id):
            """Preprocess feature and label parsed from example by parse_example function above.

            Args:
                feature ([type]): [description]
                label ([type]): [description]

            Returns:
                [type]: [description]
            """
            # add scaling phase?
            # radial_feature = complex_feature[:self.radial_feature_dim]
            # radial_feature = tf.reshape(radial_feature, (8, 8, 8))
            # radial_feature = tf.transpose(radial_feature, (1, 2, 0)) #tf.transpose(tf.reshape(radial_feature, (-1, 8, 8, 8)), (0, 3, 2, 1))

            # angular_feature = complex_feature[self.radial_feature_dim: self.radial_feature_dim + self.angular_feature_dim]
            # angular_feature = tf.reshape(angular_feature, (8, 8, 8, 64))
            # angular_feature = tf.transpose(angular_feature, [0, 2, 3, 1])
            # angular_feature = tf.reshape(angular_feature, (64, 64, 8))

            radial_feature = complex_feature[:self.radial_feature_dim]
            angular_feature = complex_feature[self.radial_feature_dim:
                                              self.radial_feature_dim + self.angular_feature_dim]

            return (radial_feature, angular_feature), label

    def generate_label(self, tfrecords, batch_size=128):
        labels_dataset = self.preprocess(tfrecords).map(
            lambda feature, labels, sample_id: labels).batch(batch_size)
        labels_array = np.concatenate([
            np.hstack([col.numpy().reshape([-1, 1]) for col in label]) for label in labels_dataset])
        labels_index = self.generate_index(tfrecords)
        labels = self.shape_preds(labels_array, index=labels_index)
        return labels

    def generate_index(self, tfrecords, batch_size=128):
        feature_names = self.create_feature_name()
        pf = self.PreprocessFunction(
            target_elements=self.feature_params['target_elements'],
            n_radial_Rs=self.feature_params['n_radial_Rs'],
            n_angular_Rs=self.feature_params['n_angular_Rs'],
            n_thetas=self.feature_params['n_thetas']
        )
        index_dataset = tfrecords.map(pf.parse_example)\
            .map(lambda feature, label, sample_id: sample_id).batch(batch_size)
        index = np.vstack([idx for idx in index_dataset]).flatten()
        index = np.vectorize(lambda bytes_obj: bytes_obj.decode())(index)
        return index

    def shape_preds(self, preds, index):
        """function for formatting prediction results into pandas.DataFrame.

        Args:
            preds ([type]): Values of the result dataframe.
            index ([type]): Index of the result dataframe.

        Returns:
            [type]: Formatted prediction result.
        """
        if isinstance(preds, list):
            shaped_preds = pd.DataFrame(np.concatenate(
                preds, axis=1), columns=self.label_colnames, index=index)
        else:
            shaped_preds = pd.DataFrame(
                preds, columns=self.label_colnames, index=index)
        return shaped_preds

    # TFRecord file generation related methods
    @staticmethod
    def _float_feature(value):
        """return float_list from float / double """
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    @staticmethod
    def _bytes_feature(value):
        """return byte_list from string / byte """
        if isinstance(value, type(tf.constant(0))):
            # BytesList won't unpack a string from an EagerTensor.
            value = value.numpy()
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    @staticmethod
    def record2example(cmp_feature, label, sample_id):
        return tf.train.Example(features=tf.train.Features(feature={
            'complex_feature': RadAngDNN._float_feature(cmp_feature),
            'label': RadAngDNN._float_feature(label),
            'sample_id': RadAngDNN._bytes_feature(sample_id)
        }))

    @staticmethod
    def _write_feature_to_tfrecords(complex_feature_files, label_file,
                                    tfr_filename, label_colnames=['pKa']):
        # load features and labels
        feature_cmp = runner.load_feature(
            complex_feature_files).astype('float32')
        label = runner.load_label(
            label_file, label_colnames=label_colnames).astype('float32')

        # process feature_cmp index
        feature_cmp.index = [idx.replace('_complex', '')
                             for idx in feature_cmp.index]

        # remove not common index
        common_idx = feature_cmp.index.intersection(label.index)
        feature_cmp = feature_cmp.loc[common_idx]
        label = label.loc[common_idx]

        # check feature
        if not all(feature_cmp.index == label.index):
            raise ValueError('feature_cmp and label have different indexes.')
        if len(feature_cmp.index) == 0:
            raise ValueError(
                'There are no features to write. Length of feature is 0.')

        # create byte type sample_id
        sample_id = np.vectorize(lambda x: x.encode())(label.index.values)

        # write data to tfr_filename
        n_sample = sample_id.shape[0]
        with tf.io.TFRecordWriter(tfr_filename) as writer:
            for i in tqdm(range(n_sample)):
                ex = RadAngDNN.record2example(
                    cmp_feature=feature_cmp.values[i],
                    label=label.values[i],
                    sample_id=sample_id[i]
                )
                writer.write(ex.SerializeToString())
        return None

    @staticmethod
    def generate_feature_tfrecords(complex_feature_files, label_file, tfr_filename, label_colnames=['pKa']):
        RadAngDNN._write_feature_to_tfrecords(
            complex_feature_files,
            label_file,
            tfr_filename,
            label_colnames=label_colnames
        )
        print(f"{tfr_filename} was saved...")
        return None


#####################################################################################################
################   Network Structure classes (use complex feature + ligand feature)    ##############
#####################################################################################################

class CNN2DCmpLgd(MyAbstractNetworkStructure):
    def __init__(self, label_colnames=['pKa_energy', 'ligand_energy'],
                 preds_colnames=['pKa_energy',
                                 'pKa_energy_by_cmp', 'ligand_energy_by_lgd'],
                 target_elements=['H', 'C', 'N', 'O', 'P', 'S', 'Cl', 'DU'], n_radial_Rs=8, n_angular_Rs=8, n_thetas=8,
                 lr=0.0001, clipnorm=1, pcc_rmse_alpha=0.7, loss_weights=[1.0, 0.5, 0.01],
                 # input_size_radial=(8, 8, 8),
                 n_filters_radial=[32, 64, 128], kernel_size_radial=3, strides_radial=1, padding_method_radial='valid',
                 maxpool_radial=False, pool_size_radial=2, maxpool_strides_radial=2, maxpool_padding_method_radial='same',
                 # input_size_angular=(64, 64, 8),
                 n_filters_angular=[32, 64, 128], kernel_size_angular=4, strides_angular=1, padding_method_angular='valid',
                 maxpool_angular=False, pool_size_angular=2, maxpool_strides_angular=2, maxpool_padding_method_angular='same',
                 n_nodes=[128, 64, 32], l2_norm=0.01, use_spectral_norm=False, dropout=0.5
                 ):
        """[summary]

        Args:
            input_size (tuple, optional): Model Parameter. [description]. Defaults to (33280,1).
            label_colnames (list, optional):  Column names of the target label. Defaults to ['pKa'].
            feature_dim (int, optional): Feature Prameter. Number of the dimension of the feature. Defaults to 33280.
            target_elements (list, optional): Feature Prameter. Element types considered in feature generation step. Defaults to ['H', 'C', 'N', 'O', 'P', 'S', 'Cl', 'DU'].
            n_radial_Rs (int, optional): Feature Prameter. Number of Rs args in radial symmetry function. Defaults to 8.
            n_angular_Rs (int, optional): Feature Prameter. Number of Rs args in angular symmetry function. Defaults to 8.
            n_thetas (int, optional): Feature Prameter. Number of theta args in angular symmetry function.  Defaults to 8.
            lr (float, optional): Model Parameter. [description]. Defaults to 0.0001.
            clipnorm (int, optional): Model Parameter. [description]. Defaults to 1.
            pcc_rmse_alpha (float, optional): Model Parameter. Ratio of PCC to RMSE in the loss function. Defaults to 0.7.
            n_filters (list, optional): Model Parameter (CNN). List of the number of the filters per layer. Defaults to [32, 64, 128].
            kernel_size (int, optional):  Model Parameter (CNN). [description]. Defaults to 4.
            strides (int, optional):  Model Parameter (CNN). [description]. Defaults to 1.
            padding_method (str, optional):  Model Parameter (CNN). [description]. Defaults to 'valid'.
            maxpool (bool, optional):  Model Parameter (CNN). description]. Defaults to False.
            pool_size (int, optional):  Model Parameter (CNN). [description]. Defaults to 2.
            maxpool_strides (int, optional):  Model Parameter (CNN). [description]. Defaults to 2.
            maxpool_padding_method (str, optional):  Model Parameter (CNN). [description]. Defaults to 'same'.
            n_nodes (list, optional):  Model Parameter (DNN). List of the number of the nodes per layer. Defaults to [128, 64, 32].
            l2_norm (float, optional): Model Parameter (DNN). [description]. Defaults to 0.01.
            use_spectral_norm (bool, optional): Model Parameter (DNN). [description]. Defaults to False.
            dropout (float, optional): Model Parameter (DNN). [description]. Defaults to 0.5.

        Returns:
            [type]: [description]
        """
        n_target_elements = len(target_elements)
        radial_feature_dim = n_radial_Rs*(n_target_elements**2)
        angular_feature_dim = n_angular_Rs*n_thetas*(n_target_elements**3)
        input_size_radial = (n_target_elements, n_radial_Rs, n_target_elements)
        input_size_angular = (
            n_target_elements*n_target_elements, n_angular_Rs*n_thetas, n_target_elements)

        self.label_colnames = label_colnames
        self.preds_colnames = preds_colnames
        self.model_params = dict(
            # input_size=input_size, # 32768/8 + 512/8  # prev ver: 576, # 4096/8 + 512/8,
            lr=lr,
            clipnorm=clipnorm,
            pcc_rmse_alpha=pcc_rmse_alpha,
            loss_weights=loss_weights,
            radial_cnn_params=dict(
                input_size=input_size_radial,
                n_filters=n_filters_radial,
                kernel_size=kernel_size_radial,
                strides=strides_radial,
                padding_method=padding_method_radial,
                maxpool=maxpool_radial,
                pool_size=pool_size_radial,
                maxpool_strides=maxpool_strides_radial,
                maxpool_padding_method=maxpool_padding_method_radial,
            ),
            angular_cnn_params=dict(
                input_size=input_size_angular,
                n_filters=n_filters_angular,
                kernel_size=kernel_size_angular,
                strides=strides_angular,
                padding_method=padding_method_angular,
                maxpool=maxpool_angular,
                pool_size=pool_size_angular,
                maxpool_strides=maxpool_strides_angular,
                maxpool_padding_method=maxpool_padding_method_angular,
            ),
            dnn_params=dict(
                n_nodes=n_nodes,
                l2_norm=l2_norm,
                use_spectral_norm=use_spectral_norm,
                dropout=dropout
            )
        )
        self.feature_params = dict(
            target_elements=target_elements,
            n_target_elements=n_target_elements,
            n_radial_Rs=n_radial_Rs,
            n_angular_Rs=n_angular_Rs,
            n_thetas=n_thetas,
            radial_feature_dim=radial_feature_dim,
            angular_feature_dim=angular_feature_dim,
            feature_dim=radial_feature_dim + angular_feature_dim,
        )
        return None

    # , target_elements, n_radial_Rs, n_angular_Rs, n_thetas):
    def create_feature_name(self):
        """Create feature names from given parameters. Feature names are required for preprocessing.

        Args:
            self ([type]): [description]

        Returns:
            feature name [list]: List of radial feature name and angular feature names
        """
        target_elements = self.feature_params['target_elements']
        n_radial_Rs = self.feature_params['n_radial_Rs']
        n_angular_Rs = self.feature_params['n_angular_Rs']
        n_thetas = self.feature_params['n_thetas']

        radial_feature_name = [f"{e_l}_{e_r}_{rs}"
                               for e_l, e_r in itertools.product(target_elements, repeat=2)
                               for rs in range(n_radial_Rs)
                               ]
        angular_feature_name = [f"{e_j}_{e_i}_{e_k}_{theta}_{rs}"
                                for e_j, e_i, e_k in itertools.product(target_elements, repeat=3)
                                for theta in range(n_thetas)
                                for rs in range(n_angular_Rs)
                                ]
        return radial_feature_name + angular_feature_name

    def create_cnn2d_layer(self, input_size=(8, 8, 8), n_filters=[32, 64, 128], kernel_size=3, strides=1, padding_method='valid', maxpool=False, pool_size=2, maxpool_strides=2, maxpool_padding_method='same'):
        model = tf.keras.Sequential()
        model.add(Conv2D(n_filters[0], kernel_size=kernel_size,
                  strides=strides, padding=padding_method, input_shape=input_size))
        for n_filter in n_filters[1:]:
            model.add(Conv2D(n_filter, kernel_size=kernel_size,
                      strides=strides, padding=padding_method))
            model.add(Activation("relu"))
            if maxpool:
                model.add(MaxPooling1D(
                    pool_size=pool_size,
                    strides=maxpool_strides,
                    padding=maxpool_padding_method,  # Padding method
                ))
        model.add(Flatten())
        return model

    def create_dnn_layer(self, n_nodes=[128, 64, 32], name='output', l2_norm=0.01, use_spectral_norm=False, dropout=0.5):
        model = tf.keras.Sequential(name=name)
        if use_spectral_norm:
            for n_node in n_nodes:
                model.add(SpectralNormalization(Dense(
                    n_node, kernel_regularizer=l2(l2_norm), activation='relu'
                ), dynamic=True))
                model.add(BatchNormalization())
                model.add(Dropout(dropout))
            model.add(SpectralNormalization(Dense(1), dynamic=True))

        else:
            for n_node in n_nodes:
                model.add(Dense(n_node, kernel_regularizer=l2(
                    l2_norm), activation='relu'))
                model.add(BatchNormalization())
                model.add(Dropout(dropout))
            model.add(Dense(1))

        return model

    def create_cnn(self):
        # lr = self.model_params['lr']
        # clipnorm = self.model_params['clipnorm']
        # pcc_rmse_alpha = self.model_params['pcc_rmse_alpha']

        radial_cnn = self.create_cnn2d_layer(
            **self.model_params['radial_cnn_params'])
        angular_cnn = self.create_cnn2d_layer(
            **self.model_params['angular_cnn_params'])

        # Input(shape=(8, 8, 8))
        input_radial = Input(
            shape=self.model_params['radial_cnn_params']['input_size'])
        input_angular = Input(
            shape=self.model_params['angular_cnn_params']['input_size'])
        x_radial = radial_cnn(input_radial)
        x_angular = angular_cnn(input_angular)
        output = concatenate([x_radial, x_angular])  # x_radial + x_angular

        model = tf.keras.models.Model(
            inputs=[input_radial, input_angular],
            outputs=[output]  # [output, output_cmp, output_lgd]
        )
        return model

    def create(self):
        lr = self.model_params['lr']
        clipnorm = self.model_params['clipnorm']
        pcc_rmse_alpha = self.model_params['pcc_rmse_alpha']
        loss_weights = self.model_params['loss_weights']

        input_cmp_radial = Input(
            shape=self.model_params['radial_cnn_params']['input_size'])
        input_cmp_angular = Input(
            shape=self.model_params['angular_cnn_params']['input_size'])
        input_lgd_radial = Input(
            shape=self.model_params['radial_cnn_params']['input_size'])
        input_lgd_angular = Input(
            shape=self.model_params['angular_cnn_params']['input_size'])

        cnn_cmp = self.create_cnn()
        cnn_lgd = self.create_cnn()

        dnn_cmp = self.create_dnn_layer(
            **self.model_params['dnn_params'], name='output_cmp')
        dnn_lgd = self.create_dnn_layer(
            **self.model_params['dnn_params'], name='output_lgd')
        dnn_joined = self.create_dnn_layer(
            **self.model_params['dnn_params'], name='output_joined')

        x_cmp = cnn_cmp([input_cmp_radial, input_cmp_angular])
        x_lgd = cnn_lgd([input_lgd_radial, input_lgd_angular])
        x_joined = concatenate([x_cmp, x_lgd])

        output_cmp = dnn_cmp(x_cmp)
        output_lgd = dnn_lgd(x_lgd)
        output_joined = dnn_joined(x_joined)

        model = tf.keras.models.Model(
            inputs=[input_cmp_radial, input_cmp_angular,
                    input_lgd_radial, input_lgd_angular],
            outputs=[output_joined, output_cmp, output_lgd]
        )

        optimizer = SGD(
            lr=lr,
            momentum=0.9,
            decay=1e-6,
            # clipnorm=clipnorm
        )

        loss_pcc_rmse = make_PCC_RMSE_with_alpha(alpha=pcc_rmse_alpha)

        model.compile(
            # loss=loss_pcc_rmse,
            # optimizer=optimizer,
            # metrics=[RMSE, PCC, loss_pcc_rmse]
            loss={
                'output_joined': loss_pcc_rmse,
                'output_cmp': loss_pcc_rmse,
                'output_lgd': loss_pcc_rmse},
            loss_weights=loss_weights,
            optimizer=optimizer,
            metrics={
                'output_joined': [RMSE, PCC, loss_pcc_rmse],
                'output_cmp': [RMSE, PCC, loss_pcc_rmse],
                'output_lgd': [RMSE, PCC, loss_pcc_rmse],
            }
        )

        return model

    def compile(self, model):
        """function for compiling the network.

        Args:
            network ([type]): [description]

        Returns:
            [type]: [description]
        """
        clipnorm = self.model_params['clipnorm']
        pcc_rmse_alpha = self.model_params['pcc_rmse_alpha']
        lr = self.model_params['lr']
        loss_weights = self.model_params['loss_weights']

        optimizer = SGD(
            lr=lr,
            momentum=0.9,
            decay=1e-6,
            # clipnorm=clipnorm
        )

        loss_pcc_rmse = make_PCC_RMSE_with_alpha(alpha=pcc_rmse_alpha)

        model.compile(
            # loss=loss_pcc_rmse,
            # optimizer=optimizer,
            # metrics=[RMSE, PCC, loss_pcc_rmse]
            loss={
                'output_joined': loss_pcc_rmse,
                'output_cmp': loss_pcc_rmse,
                'output_lgd': loss_pcc_rmse},
            loss_weights=loss_weights,
            optimizer=optimizer,
            metrics={
                'output_joined': [RMSE, PCC, loss_pcc_rmse],
                'output_cmp': [RMSE, PCC, loss_pcc_rmse],
                'output_lgd': [RMSE, PCC, loss_pcc_rmse],
            }
        )
        return model

    def _preprocess_pandas(self, X, y):
        pf = self.PreprocessFunction(
            target_elements=self.feature_params['target_elements'],
            n_radial_Rs=self.feature_params['n_radial_Rs'],
            n_angular_Rs=self.feature_params['n_angular_Rs'],
            n_thetas=self.feature_params['n_thetas'],
        )
        processed_feature, processed_label = pf.process_pandas(X, y)
        return processed_feature, processed_label

    def _preprocess_tfrecord(self, tfrecords):
        """Function to preprocess input of data type TFRecords.

        Args:
            tfrecords ([TFRecordDatasetV2]): Input and target data. 

        Returns:
            [TFRecordDatasetV2]: Pre-processed input and target data.
        """
        # feature_names = self.create_feature_name()
        pf = self.PreprocessFunction(
            # radial_feature_dim=self.feature_params['radial_feature_dim'],
            # angular_feature_dim=self.feature_params['angular_feature_dim'],
            target_elements=self.feature_params['target_elements'],
            n_radial_Rs=self.feature_params['n_radial_Rs'],
            n_angular_Rs=self.feature_params['n_angular_Rs'],
            n_thetas=self.feature_params['n_thetas']
        )
        return tfrecords.map(pf.parse_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
            .map(pf.process_tfrecord, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    def preprocess(self, X, y=None):
        """Pre-process input and target data into a form acceptable to the model.

        Args:
            X : Input data. Pandas DataFrame of TFRecordDatasetV2 can be accepted.
            y ([type], optional): Target data. If the type of X is TFRecordDatasetV2 then y arg is ignored. Defaults to None.

        Returns:
            [type]: Pre-processed input and target data
        """
        if isinstance(X, TFRecordDataset):
            return self._preprocess_tfrecord(X)
        elif isinstance(X, tuple):  # isinstance(X, pd.core.frame.DataFrame):
            return self._preprocess_pandas(X, y)
        else:
            TypeError('''Feature(X) and label(y) must be pandas.DataFrame or TFRecordDataset.
            Please check the type of your data.''')

    class PreprocessFunction(object):
        """Class for creating functions for preprocessing input and target data.
        Specializes in preprocessing CNN1D class input data.
        In the CNN1D class, the feature names are very important because the feature partitioning depends on the names
        This class determines the element of the ligand from the feature name using regular expressions and divide the feature by the element of the ligand.

        Args:
            object ([type]): [description]
        """

        def __init__(self,  # radial_feature_dim=512, angular_feature_dim=32768,
                     target_elements=['H', 'C', 'N',
                                      'O', 'P', 'S', 'Cl', 'DU'],
                     n_radial_Rs=8, n_angular_Rs=8, n_thetas=8,
                     ):
            """
            In the CNN1D class, the feature names are very important because the feature partitioning depends on the names.
            This class determines the element of the ligand from the feature name using regular expressions and divide the feature by the element of the ligand.
            Therefore, this class needs feature_names, target_elements and regex_template.

            Args:
                feature_names ([list, pd.core.strings.StringMethods]): feature names.
                target_elements ([list]): Element types to be considered. Default to ['H', 'C', 'N', 'O', 'P', 'S', 'Cl', 'DU'].
            """
            # self.complex_feature_names = pd.Index(complex_feature_names).str \
            #     if not isinstance(complex_feature_names, pd.core.strings.StringMethods) else complex_feature_names
            self.target_elements = target_elements
            self.n_target_elements = len(target_elements)
            self.n_radial_Rs = n_radial_Rs
            self.n_angular_Rs = n_angular_Rs
            self.n_thetas = n_thetas
            self.radial_feature_dim = self.n_target_elements**2 * \
                n_radial_Rs  # radial_feature_dim
            self.angular_feature_dim = self.n_target_elements**3 * \
                n_angular_Rs * n_thetas  # angular_feature_dim
            self.complex_feature_dim = self.radial_feature_dim + self.angular_feature_dim
            self.ligand_feature_dim = self.radial_feature_dim + self.angular_feature_dim
            # self.target_elements = target_elements #['H', 'C', 'N', 'O', 'P', 'S', 'Cl', 'DU']
            # self.regex_template = "(^\w?\w?_{element}_\w?\w?_\d+_\d+$)|(^{element}_\w?\w?_\d+$)"
            return None

        # TODO: pandas is not yet supported.
        def process_pandas(self, feature, label=None):
            """Preprocess feature and label of data type pandas DataFrame.

            Args:
                feature ([tuple of pandas.DataFrames]): Input data. Tuple of complex feature (pandas df) and ligand feature (pandas df)
                label ([pandas.core.frame.DataFrame], optional): Target data. Defaults to None.

            Returns:
                processed_feature [pandas.core.frame.DataFrame]: Processed input data.
                processed_label [pandas.core.frame.DataFrame]: Processed target data.
            """
            complex_feature, ligand_feature = feature

            # Process Complex feature
            cmp_radial_feature = complex_feature.values[:,
                                                        :self.radial_feature_dim]
            cmp_radial_feature = tf.reshape(cmp_radial_feature,
                                            (-1, self.n_target_elements,
                                             self.n_target_elements, self.n_radial_Rs)
                                            )
            cmp_radial_feature = tf.transpose(cmp_radial_feature, (0, 2, 3, 1))

            cmp_angular_feature = complex_feature.values[:,
                                                         self.radial_feature_dim: self.radial_feature_dim + self.angular_feature_dim]
            cmp_angular_feature = tf.reshape(cmp_angular_feature,
                                             (-1, self.n_target_elements, self.n_target_elements, self.n_target_elements, self.n_angular_Rs*self.n_thetas))
            cmp_angular_feature = tf.transpose(
                cmp_angular_feature, [0, 1, 3, 4, 2])
            cmp_angular_feature = tf.reshape(cmp_angular_feature,
                                             (-1, self.n_target_elements*self.n_target_elements,
                                              self.n_angular_Rs*self.n_thetas, self.n_target_elements)
                                             )

            lgd_radial_feature = ligand_feature.values[:,
                                                       :self.radial_feature_dim]
            lgd_radial_feature = tf.reshape(lgd_radial_feature,
                                            (-1, self.n_target_elements,
                                             self.n_target_elements, self.n_radial_Rs)
                                            )
            lgd_radial_feature = tf.transpose(lgd_radial_feature, (0, 2, 3, 1))

            lgd_angular_feature = ligand_feature.values[:,
                                                        self.radial_feature_dim: self.radial_feature_dim + self.angular_feature_dim]
            lgd_angular_feature = tf.reshape(lgd_angular_feature,
                                             (-1, self.n_target_elements, self.n_target_elements, self.n_target_elements, self.n_angular_Rs*self.n_thetas))
            lgd_angular_feature = tf.transpose(
                lgd_angular_feature, [0, 1, 3, 4, 2])
            lgd_angular_feature = tf.reshape(lgd_angular_feature,
                                             (-1, self.n_target_elements*self.n_target_elements,
                                              self.n_angular_Rs*self.n_thetas, self.n_target_elements)
                                             )

            complex_label = label.values[:, 0].astype(
                "float64").copy() if label is not None else None
            ligand_label = label.values[:, 1].astype(
                "float64").copy() if label is not None else None

            return (cmp_radial_feature, cmp_angular_feature, lgd_radial_feature, lgd_angular_feature), (complex_label, complex_label, ligand_label)

        @tf.autograph.experimental.do_not_convert
        def parse_example(self, example):
            """parse tfrecord to feature and label.

            Args:
                example ([tensorflow.python.framework.ops.Tensor]): [description]

            Returns:
                x [tensorflow.python.framework.ops.Tensor]: [description]
                y [tensorflow.python.framework.ops.Tensor]: [description]
            """
            # feature_dim = 33280
            features = tf.io.parse_single_example(
                example,
                features={
                    'complex_feature': tf.io.FixedLenFeature([self.complex_feature_dim], dtype=tf.float32),
                    'ligand_feature': tf.io.FixedLenFeature([self.ligand_feature_dim], dtype=tf.float32),
                    'label': tf.io.FixedLenFeature([2], dtype=tf.float32),
                    'sample_id': tf.io.FixedLenFeature([1], dtype=tf.string)
                }
            )
            complex_feature = features["complex_feature"]
            ligand_feature = features["ligand_feature"]
            label = features["label"]
            sample_id = features["sample_id"]

            return complex_feature, ligand_feature, label, sample_id

        @tf.autograph.experimental.do_not_convert
        def process_tfrecord(self, complex_feature, ligand_feature, label, sample_id):
            """Preprocess feature and label parsed from example by parse_example function above.

            Args:
                feature ([type]): [description]
                label ([type]): [description]

            Returns:
                [type]: [description]
            """
            # add scaling phase?
            # radial_feature = complex_feature[:self.radial_feature_dim]
            # radial_feature = tf.reshape(radial_feature, (8, 8, 8))
            # radial_feature = tf.transpose(radial_feature, (1, 2, 0)) #tf.transpose(tf.reshape(radial_feature, (-1, 8, 8, 8)), (0, 3, 2, 1))

            # angular_feature = complex_feature[self.radial_feature_dim: self.radial_feature_dim + self.angular_feature_dim]
            # angular_feature = tf.reshape(angular_feature, (8, 8, 8, 64))
            # angular_feature = tf.transpose(angular_feature, [0, 2, 3, 1])
            # angular_feature = tf.reshape(angular_feature, (64, 64, 8))

            # Process Complex feature
            cmp_radial_feature = complex_feature[:self.radial_feature_dim]
            cmp_radial_feature = tf.reshape(cmp_radial_feature,
                                            (self.n_target_elements,
                                             self.n_target_elements, self.n_radial_Rs)
                                            )
            cmp_radial_feature = tf.transpose(cmp_radial_feature, (1, 2, 0))

            cmp_angular_feature = complex_feature[self.radial_feature_dim:
                                                  self.radial_feature_dim + self.angular_feature_dim]
            cmp_angular_feature = tf.reshape(cmp_angular_feature,
                                             (self.n_target_elements, self.n_target_elements, self.n_target_elements, self.n_angular_Rs*self.n_thetas))
            cmp_angular_feature = tf.transpose(
                cmp_angular_feature, [0, 2, 3, 1])
            cmp_angular_feature = tf.reshape(cmp_angular_feature,
                                             (self.n_target_elements*self.n_target_elements,
                                              self.n_angular_Rs*self.n_thetas, self.n_target_elements)
                                             )

            # Process Ligand feature
            lgd_radial_feature = ligand_feature[:self.radial_feature_dim]
            lgd_radial_feature = tf.reshape(lgd_radial_feature,
                                            (self.n_target_elements,
                                             self.n_target_elements, self.n_radial_Rs)
                                            )
            lgd_radial_feature = tf.transpose(lgd_radial_feature, (1, 2, 0))

            lgd_angular_feature = ligand_feature[self.radial_feature_dim:
                                                 self.radial_feature_dim + self.angular_feature_dim]
            lgd_angular_feature = tf.reshape(lgd_angular_feature,
                                             (self.n_target_elements, self.n_target_elements, self.n_target_elements, self.n_angular_Rs*self.n_thetas))
            lgd_angular_feature = tf.transpose(
                lgd_angular_feature, [0, 2, 3, 1])
            lgd_angular_feature = tf.reshape(lgd_angular_feature,
                                             (self.n_target_elements*self.n_target_elements,
                                              self.n_angular_Rs*self.n_thetas, self.n_target_elements)
                                             )

            complex_label = label[0]
            ligand_label = label[1]
            return (cmp_radial_feature, cmp_angular_feature, lgd_radial_feature, lgd_angular_feature), (complex_label, complex_label, ligand_label)
            # return (processed_complex_feature, processed_lgdand_feature),  (complex_label, complex_label, ligand_label)

    def generate_label(self, tfrecords, batch_size=128):
        pf = self.PreprocessFunction(
            target_elements=self.feature_params['target_elements'],
            n_radial_Rs=self.feature_params['n_radial_Rs'],
            n_angular_Rs=self.feature_params['n_angular_Rs'],
            n_thetas=self.feature_params['n_thetas']
        )
        labels_dataset = tfrecords.map(pf.parse_example)\
            .map(lambda complex_feature, ligand_feature, labels, sample_id: labels).batch(batch_size)
        labels_array = np.concatenate([label for label in labels_dataset])
        labels_index = self.generate_index(tfrecords)
        labels = self.shape_preds(labels_array, index=labels_index)
        return labels

    def generate_index(self, tfrecords, batch_size=128):
        # feature_names = self.create_feature_name()
        pf = self.PreprocessFunction(
            target_elements=self.feature_params['target_elements'],
            n_radial_Rs=self.feature_params['n_radial_Rs'],
            n_angular_Rs=self.feature_params['n_angular_Rs'],
            n_thetas=self.feature_params['n_thetas']
        )
        index_dataset = tfrecords.map(pf.parse_example)\
            .map(lambda complex_feature, ligand_feature, label, sample_id: sample_id).batch(batch_size)
        index = np.vstack([idx for idx in index_dataset]).flatten()
        index = np.vectorize(lambda bytes_obj: bytes_obj.decode())(index)
        return index

    def shape_preds(self, preds, index):
        """function for formatting prediction results into pandas.DataFrame.

        Args:
            preds ([type]): Values of the result dataframe.
            index ([type]): Index of the result dataframe.

        Returns:
            [type]: Formatted prediction result.
        """
        if isinstance(preds, list):
            shaped_preds = pd.DataFrame(np.concatenate(
                preds, axis=1), columns=self.preds_colnames, index=index)
        else:
            shaped_preds = pd.DataFrame(
                preds, columns=self.preds_colnames, index=index)
        return shaped_preds

    # TFRecord file generation related methods
    @staticmethod
    def _float_feature(value):
        """return float_list from float / double """
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    @staticmethod
    def _bytes_feature(value):
        """return byte_list from string / byte """
        if isinstance(value, type(tf.constant(0))):
            # BytesList won't unpack a string from an EagerTensor.
            value = value.numpy()
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    @staticmethod
    def record2example(cmp_feature, lgd_feature, label, sample_id):
        return tf.train.Example(features=tf.train.Features(feature={
            'complex_feature': CNN2DCmpLgd._float_feature(cmp_feature),
            'ligand_feature': CNN2DCmpLgd._float_feature(lgd_feature),
            'label': CNN2DCmpLgd._float_feature(label),
            'sample_id': CNN2DCmpLgd._bytes_feature(sample_id)
        }))

    @staticmethod
    def _write_feature_to_tfrecords(complex_feature_files, ligand_feature_files, label_file,
                                    tfr_filename, label_colnames=[
                                        'pKa_energy', 'ligand_energy'],
                                    scheduler='processes'):
        # load features and labels
        feature_cmp = runner.load_feature(
            complex_feature_files, scheduler=scheduler).astype('float32')
        feature_lgd = runner.load_feature(
            ligand_feature_files, scheduler=scheduler).astype('float32')
        label = runner.load_label(
            label_file, label_colnames=label_colnames).astype('float32')

        # remove not common index
        common_idx = feature_cmp.index.intersection(label.index)
        feature_cmp = feature_cmp.loc[common_idx]
        feature_lgd = feature_lgd.loc[common_idx]

        label = label.loc[common_idx]

        if not all(feature_cmp.index == label.index):
            raise ValueError('feature_cmp and label have different indexes.')
        elif not all(feature_lgd.index == label.index):
            raise ValueError('feature_lgd and label have different indexes.')

        # check whether len of feature is not 0
        if len(feature_cmp.index) == 0 or len(feature_lgd.index) == 0:
            raise ValueError(
                'There are no features to write. Length of feature is 0.')

        # create byte type sample_id
        sample_id = np.vectorize(lambda x: x.encode())(label.index.values)

        # write data to tfr_filename
        n_sample = sample_id.shape[0]
        with tf.io.TFRecordWriter(tfr_filename) as writer:
            for i in tqdm(range(n_sample)):
                ex = CNN2DCmpLgd.record2example(
                    cmp_feature=feature_cmp.values[i],
                    lgd_feature=feature_lgd.values[i],
                    label=label.values[i],
                    sample_id=sample_id[i]
                )
                writer.write(ex.SerializeToString())
        return None

    @staticmethod
    def generate_feature_tfrecords(complex_feature_files,
                                   ligand_feature_files,
                                   label_file,
                                   tfr_filename,
                                   label_colnames=[
                                       'pKa_energy', 'ligand_energy'],
                                   scheduler='processes'):

        CNN2DCmpLgd._write_feature_to_tfrecords(
            complex_feature_files,
            ligand_feature_files,
            label_file,
            tfr_filename,
            label_colnames=label_colnames,
            scheduler=scheduler
        )
        print(f"{tfr_filename} was saved...")
        return None


class ComplexNNplusSubLigandNN(MyAbstractNetworkStructure):
    def __init__(self, label_colnames=['pKa_energy', 'pKa_energy_by_cmp', 'ligand_energy'],
                 input_size_per_element=4160,  # 32768/8 + 512/8  # prev ver: 576, # 4096/8 + 512/8,
                 n_layers=2, n_nodes=64, output_size_per_element=1,
                 lr=0.0001, clipnorm=1, output_layer_style='sum', pcc_rmse_alpha=0.7,
                 use_spectral_norm=False, l2_norm=0.01, dropout=0.5, dropout_input=0.2):
        """Instantiate a model with the specified parameters

        Args:
            label_colnames (list, optional): Column names of the target label. Defaults to ['pKa_energy'].
            input_size_per_element (int): Default to 4160 (32768/8 + 512/8).
            target_elements (list): Element types to be considered. Default to ['H', 'C', 'N', 'O', 'P', 'S', 'Cl', 'DU'].
            n_target_elements (int): Number of element types to be considered. Default to 8.
            n_radial_Rs (int): Number of Rs args in radial symmetry function. Default to 8.
            n_angular_Rs (int): Number of Rs args in angular symmetry function. Default to 8.
            n_thetas (int): Number of theta args in angular symmetry function. Default to 8.
            n_layers (int): Number of layers per network for each element type.
            n_nodes (int): Number of layers per layer.
            lr (float): Learning rate. Default to 0.0001.
            clipnorm (float): Clip norm value. Default to 1. *Setting clipnorm or clipvalue is currently unsupported when using a distribution strategy (MirroredStrategy).
            output_layer_style (str): Style for summarizing the output from the each element type networks . 'sum', 'fc' or 'sum+fc' is acceptable. Default to 'sum'.
            use_spectral_norm (bool): Whethere to use spectral normalization to Dense layers. Default to True.
            pcc_rmse_alpha (float): Ratio of PCC to RMSE in the loss function. Default to 0.7.

        Returns:
            [type]: [description]
        """
        self.label_colnames = label_colnames
        # default parameters of model_params
        self.model_params = dict(
            # 32768/8 + 512/8  # prev ver: 576, # 4096/8 + 512/8,
            input_size_per_element=input_size_per_element,
            n_layers=n_layers,
            n_nodes=n_nodes,
            output_size_per_element=output_size_per_element,
            lr=lr,
            clipnorm=clipnorm,
            output_layer_style=output_layer_style,
            use_spectral_norm=use_spectral_norm,
            pcc_rmse_alpha=pcc_rmse_alpha,
            l2_norm=l2_norm,
            dropout=dropout,
            dropout_input=dropout_input
        )

        self.feature_params = dict(
            target_elements=['H', 'C', 'N', 'O', 'P', 'S', 'Cl', 'DU'],
            n_target_elements=8,
            n_radial_Rs=8,
            n_angular_Rs=8,
            n_thetas=8
        )
        return None

    # , target_elements, n_radial_Rs, n_angular_Rs, n_thetas):
    def create_feature_name(self):
        """Create feature names from given parameters. Feature names are required for preprocessing.

        Args:
            self ([type]): [description]

        Returns:
            feature name [list]: List of radial feature name and angular feature names
        """
        target_elements = self.feature_params['target_elements']
        n_radial_Rs = self.feature_params['n_radial_Rs']
        n_angular_Rs = self.feature_params['n_angular_Rs']
        n_thetas = self.feature_params['n_thetas']

        radial_feature_name = [f"{e_l}_{e_r}_{rs}"
                               for e_l, e_r in itertools.product(target_elements, repeat=2)
                               for rs in range(n_radial_Rs)
                               ]
        angular_feature_name = [f"{e_j}_{e_i}_{e_k}_{theta}_{rs}"
                                for e_j, e_i, e_k in itertools.product(target_elements, repeat=3)
                                for theta in range(n_thetas)
                                for rs in range(n_angular_Rs)
                                ]
        return radial_feature_name + angular_feature_name

    def create_variable_network(self, n_layers, n_nodes, input_size=576, output_size=1, use_spectral_norm=True, l2_norm=0.01, dropout=0.5, dropout_input=0.2):
        """Function to create a network with a variable number of layers and nodes.

        Args:
            n_layers ([type]): [description]
            n_nodes ([type]): [description]
            input_size (int, optional): [description]. Defaults to 576.
            use_spectral_norm (bool, optional): [description]. Defaults to True.

        Returns:
            network ['tensorflow.python.keras.engine.sequential.Sequential']: tensorflow sequential model with the specified number of layers and nodes.
        """
        network = tf.keras.Sequential()
        network.add(tf.keras.Input(shape=input_size))
        network.add(Dropout(dropout_input))

        if use_spectral_norm:
            for _ in range(n_layers):
                network.add(SpectralNormalization(
                    Dense(n_nodes, kernel_regularizer=l2(l2_norm), activation='relu'), dynamic=True))
                # network.add(Activation("relu"))
                network.add(BatchNormalization())
                network.add(Dropout(dropout))
            network.add(SpectralNormalization(
                Dense(output_size), dynamic=True))
        else:
            for _ in range(n_layers):
                network.add(Dense(n_nodes, kernel_regularizer=l2(
                    l2_norm), activation='relu'))
                # network.add(Activation("relu"))
                network.add(BatchNormalization())
                network.add(Dropout(dropout))
            network.add(Dense(output_size))

        return network

    def create_output_layer(self, output_layer_style, use_spectral_norm, l2_norm=0.01, dropout=0.5, name='output', n_inputs=None, input_size=None):
        # n_target_elements = self.feature_params['n_target_elements']
        # output_size_per_element = self.model_params['output_size_per_element']
        # if n_inputs is None:
        #     n_inputs = self.feature_params['n_target_elements']

        # if input_size is None:
        #     input_size = self.model_params['output_size_per_element']

        inputs = [Input(shape=input_size) for _ in range(n_inputs)]

        if output_layer_style == 'sum':
            output = tf.keras.layers.Add()(inputs)

        elif output_layer_style == 'fc':
            x = concatenate(inputs)
            x = Dropout(dropout)(x)

            if l2_norm != 0.0:
                for n_node in [128, 64, 32]:
                    x = SpectralNormalization(Dense(n_node, kernel_regularizer=l2(l2_norm), activation='relu'), dynamic=True)(x) \
                        if use_spectral_norm else Dense(n_node, kernel_regularizer=l2(l2_norm), activation='relu')(x)
                    x = Dropout(dropout)(x)
            elif l2_norm == 0.0:
                for n_node in [128, 64, 32]:
                    x = SpectralNormalization(Dense(n_node, activation='relu'), dynamic=True)(x) \
                        if use_spectral_norm else Dense(n_node, activation='relu')(x)
                    x = Dropout(dropout)(x)

            output = SpectralNormalization(Dense(1), dynamic=True)(x) \
                if use_spectral_norm else Dense(1)(x)
        elif output_layer_style == 'sum+fc':
            x = tf.keras.layers.Add()(inputs)
            if l2_norm != 0.0:
                for n_node in [128, 64, 32]:
                    x = SpectralNormalization(Dense(n_node, kernel_regularizer=l2(l2_norm), activation='relu'), dynamic=True)(x) \
                        if use_spectral_norm else Dense(n_node, kernel_regularizer=l2(l2_norm), activation='relu')(x)
                    x = Dropout(dropout)(x)
            elif l2_norm == 0.0:
                for n_node in [128, 64, 32]:
                    x = SpectralNormalization(Dense(n_node, activation='relu'), dynamic=True)(x) \
                        if use_spectral_norm else Dense(n_node, activation='relu')(x)
                    x = Dropout(dropout)(x)

            output = SpectralNormalization(Dense(1), dynamic=True)(x)\
                if use_spectral_norm else Dense(1)(x)
        else:
            raise ValueError('output_layer_style must be sum, fc or sum+fc')

        network = tf.keras.models.Model(
            inputs=[inputs],
            outputs=[output],
            name=name
        )

        # create output layer -------------------------------------------------------
        # if output_layer_style == 'sum':
        #     output = tf.keras.layers.Add(name='output')(outputs)
        # elif output_layer_style == 'fc':
        #     x = concatenate(outputs)
        #     x = SpectralNormalization(Dense(256, kernel_regularizer=l2(l2_norm), activation='relu'), dynamic=True)(x) \
        #         if use_spectral_norm else Dense(256, kernel_regularizer=l2(l2_norm), activation='relu')(x)
        #     output = SpectralNormalization(Dense(1), dynamic=True, name='output')(x) \
        #         if use_spectral_norm else Dense(1, name='output')(x)
        # elif output_layer_style == 'sum+fc':
        #     x = tf.keras.layers.Add()(outputs)
        #     x = SpectralNormalization(Dense(256, kernel_regularizer=l2(l2_norm), activation='relu'), dynamic=True)(x)\
        #         if use_spectral_norm else Dense(256, kernel_regularizer=l2(l2_norm), activation='relu')(x)
        #     output = SpectralNormalization(Dense(1), dynamic=True, name='output')(x)\
        #         if use_spectral_norm else Dense(1, name='output')(x)
        # else:
        #     raise ValueError('output_layer_style must be sum, fc or sum+fc')

        return network

    def create_elementwise_dnn(self, name='elementwise_dnn'):
        """Create tensorflow model. 

        Raises:
            ValueError: Raised if output_layer_style is not sum, fc or sum+fc.

        Returns:
            network [tensorflow.python.keras.engine.training.Model]: A model that has networks for each element type.
        """
        # set params -------------------------------------------------------
        n_target_elements = self.feature_params['n_target_elements']
        input_size_per_element = self.model_params['input_size_per_element']
        lr = self.model_params['lr']
        clipnorm = self.model_params['clipnorm']
        n_layers = self.model_params['n_layers']
        n_nodes = self.model_params['n_nodes']
        output_layer_style = self.model_params['output_layer_style']
        use_spectral_norm = self.model_params['use_spectral_norm']
        output_size_per_element = self.model_params['output_size_per_element']
        l2_norm = self.model_params['l2_norm']
        dropout = self.model_params['dropout']
        dropout_input = self.model_params['dropout_input']

        # create middle layers -------------------------------------------------------
        inputs = [Input(shape=input_size_per_element)
                  for _ in range(n_target_elements)]

        elementwise_models = [
            self.create_variable_network(
                n_layers, n_nodes,
                input_size=input_size_per_element,
                output_size=output_size_per_element,
                use_spectral_norm=use_spectral_norm,
                l2_norm=l2_norm,
                dropout=dropout,
                dropout_input=dropout_input
            )
            for _ in range(n_target_elements)
        ]

        outputs = [model(input)
                   for model, input in zip(elementwise_models, inputs)]

        # create output layer -------------------------------------------------------
        # if output_layer_style == 'sum':
        #     output = tf.keras.layers.Add(name='output')(outputs)
        # elif output_layer_style == 'fc':
        #     x = concatenate(outputs)
        #     x = SpectralNormalization(Dense(256, kernel_regularizer=l2(l2_norm), activation='relu'), dynamic=True)(x) \
        #         if use_spectral_norm else Dense(256, kernel_regularizer=l2(l2_norm), activation='relu')(x)
        #     output = SpectralNormalization(Dense(1), dynamic=True, name='output')(x) \
        #         if use_spectral_norm else Dense(1, name='output')(x)
        # elif output_layer_style == 'sum+fc':
        #     x = tf.keras.layers.Add()(outputs)
        #     x = SpectralNormalization(Dense(256, kernel_regularizer=l2(l2_norm), activation='relu'), dynamic=True)(x)\
        #         if use_spectral_norm else Dense(256, kernel_regularizer=l2(l2_norm), activation='relu')(x)
        #     output = SpectralNormalization(Dense(1), dynamic=True, name='output')(x)\
        #         if use_spectral_norm else Dense(1, name='output')(x)
        # else:
        #     raise ValueError('output_layer_style must be sum, fc or sum+fc')

        # output_layer = self.create_output_layer(output_layer_style, use_spectral_norm)
        # output_layer(outputs)

        # instantiate the model -------------------------------------------------------
        network = tf.keras.models.Model(
            inputs=[inputs],
            outputs=outputs,
            name=name
        )
        return network

    def create(self):
        n_target_elements = self.feature_params['n_target_elements']
        input_size_per_element = self.model_params['input_size_per_element']
        output_layer_style = self.model_params['output_layer_style']
        use_spectral_norm = self.model_params['use_spectral_norm']
        output_size_per_element = self.model_params['output_size_per_element']
        lr = self.model_params['lr']
        clipnorm = self.model_params['clipnorm']
        l2_norm = self.model_params['l2_norm']
        dropout = self.model_params['dropout']
        dropout_input = self.model_params['dropout_input']

        inputs_cmp = [Input(shape=input_size_per_element)
                      for _ in range(n_target_elements)]
        inputs_lgd = [Input(shape=input_size_per_element)
                      for _ in range(n_target_elements)]

        cmp_feature_network = self.create_elementwise_dnn(
            name='cmp_feature_network')
        lgd_feature_network = self.create_elementwise_dnn(
            name='lgd_feature_network')
        cmp_output_layer = self.create_output_layer(
            output_layer_style, use_spectral_norm, name='output_cmp',
            n_inputs=n_target_elements, input_size=output_size_per_element,
            l2_norm=l2_norm, dropout=dropout)
        lgd_output_layer = self.create_output_layer(
            output_layer_style, use_spectral_norm, name='output_lgd',
            n_inputs=n_target_elements, input_size=output_size_per_element,
            l2_norm=0, dropout=dropout)
        joined_output_layer = self.create_output_layer(
            output_layer_style, use_spectral_norm, name='output',
            n_inputs=n_target_elements*2, input_size=output_size_per_element,
            l2_norm=l2_norm, dropout=dropout)

        x_cmp = cmp_feature_network(inputs_cmp)
        x_lgd = lgd_feature_network(inputs_lgd)
        # concatenate([concatenate(x_cmp), concatenate(x_lgd)])
        x_joined = x_cmp + x_lgd

        output_cmp = cmp_output_layer(x_cmp)
        output_lgd = lgd_output_layer(x_lgd)
        output = joined_output_layer(x_joined)

        network = tf.keras.models.Model(
            inputs=[inputs_cmp, inputs_lgd],
            # [output, output_cmp, output_lgd]
            outputs=[output, output_cmp, output_lgd]
        )

        optimizer = SGD(
            lr=lr,
            momentum=0.9,
            decay=1e-6,
            # clipnorm=clipnorm
        )

        loss_pcc_rmse = make_PCC_RMSE_with_alpha(
            alpha=self.model_params['pcc_rmse_alpha'])

        network.compile(
            loss={
                'output': loss_pcc_rmse,
                'output_cmp': loss_pcc_rmse,
                'output_lgd': loss_pcc_rmse},
            loss_weights=[1.0, 0.5, 0.01],
            optimizer=optimizer,
            metrics={
                'output': [RMSE, PCC, loss_pcc_rmse],
                'output_cmp': [RMSE, PCC, loss_pcc_rmse],
                'output_lgd': [RMSE, PCC, loss_pcc_rmse],
            }
        )

        return network

    def _preprocess_pandas(self, X, y):
        return None

    def _preprocess_tfrecord(self, tfrecords):
        """Function to preprocess input of data type TFRecords.

        Args:
            tfrecords ([TFRecordDatasetV2]): Input and target data. 

        Returns:
            [TFRecordDatasetV2]: Pre-processed input and target data.
        """
        feature_names = self.create_feature_name()
        pf = self.PreprocessFunction(
            complex_feature_names=feature_names,
            ligand_feature_names=feature_names,
            target_elements=self.feature_params['target_elements']
        )
        return tfrecords.map(pf.parse_example).map(pf.process_tfrecord)

    def preprocess(self, X, y=None):
        """Pre-process input and target data into a form acceptable to the model.

        Args:
            X : Input data. Pandas DataFrame of TFRecordDatasetV2 can be accepted.
            y ([type], optional): Target data. If the type of X is TFRecordDatasetV2 then y arg is ignored. Defaults to None.

        Returns:
            [type]: Pre-processed input and target data
        """
        if isinstance(X, TFRecordDataset):
            return self._preprocess_tfrecord(X)
        elif isinstance(X, pd.core.frame.DataFrame):
            return self._preprocess_pandas(X, y)
        else:
            TypeError('''Feature(X) and label(y) must be pandas.DataFrame or TFRecordDataset.
            Please check the type of your data.''')

    class PreprocessFunction(object):
        """Class for creating functions for preprocessing input and target data.
        Specializes in preprocessing ComplexNNplusSubLigandNN class input data.
        In the ComplexNNplusSubLigandNN class, the feature names are very important because the feature partitioning depends on the names
        This class determines the element of the ligand from the feature name using regular expressions and divide the feature by the element of the ligand.

        Args:
            object ([type]): [description]
        """

        def __init__(self, complex_feature_names, ligand_feature_names, target_elements=['H', 'C', 'N', 'O', 'P', 'S', 'Cl', 'DU']):
            """instantiate by passing feature name. 
            In the ComplexNNplusSubLigandNN class, the feature names are very important because the feature partitioning depends on the names.
            This class determines the element of the ligand from the feature name using regular expressions and divide the feature by the element of the ligand.
            Therefore, this class needs feature_names, target_elements and regex_template.

            Args:
                feature_names ([list, pd.core.strings.StringMethods]): feature names.
                target_elements ([list]): Element types to be considered. Default to ['H', 'C', 'N', 'O', 'P', 'S', 'Cl', 'DU'].
            """
            self.complex_feature_names = pd.Index(complex_feature_names).str \
                if not isinstance(complex_feature_names, pd.core.strings.StringMethods) else complex_feature_names
            self.ligand_feature_names = pd.Index(ligand_feature_names).str \
                if not isinstance(ligand_feature_names, pd.core.strings.StringMethods) else ligand_feature_names
            self.complex_feature_dim = self.complex_feature_names.__dict__[
                '_orig'].size
            self.ligand_feature_dim = self.ligand_feature_names.__dict__[
                '_orig'].size
            # ['H', 'C', 'N', 'O', 'P', 'S', 'Cl', 'DU']
            self.target_elements = target_elements
            self.regex_template = "(^\w?\w?_{element}_\w?\w?_\d+_\d+$)|(^{element}_\w?\w?_\d+$)"

        def process_pandas(self, feature, label=None):
            """Preprocess feature and label of data type pandas DataFrame.

            Args:
                feature ([pandas.core.frame.DataFrame]): Input data.
                label ([pandas.core.frame.DataFrame], optional): Target data. Defaults to None.

            Returns:
                processed_feature [pandas.core.frame.DataFrame]: Processed input data.
                processed_label [pandas.core.frame.DataFrame]: Processed target data.
            """
            processed_feature = [
                feature.loc[:, self.feature_names.match(
                    self.regex_template.format(element=element))].astype("float64")
                for element in self.target_elements
            ]
            processed_label = label.values[:, 0].astype(
                "float64").copy() if label is not None else None
            return processed_feature, processed_label

        # @staticmethod
        def parse_example(self, example):
            """parse tfrecord to feature and label.

            Args:
                example ([tensorflow.python.framework.ops.Tensor]): [description]

            Returns:
                x [tensorflow.python.framework.ops.Tensor]: [description]
                y [tensorflow.python.framework.ops.Tensor]: [description]
            """
            # feature_dim = 33280
            features = tf.io.parse_single_example(
                example,
                features={
                    # tf.float3
                    'complex_feature': tf.io.FixedLenFeature([self.complex_feature_dim], dtype=tf.float32),
                    'ligand_feature': tf.io.FixedLenFeature([self.ligand_feature_dim], dtype=tf.float32),
                    'label': tf.io.FixedLenFeature([2], dtype=tf.float32),
                    'sample_id': tf.io.FixedLenFeature([1], dtype=tf.string)
                }
            )
            complex_feature = features["complex_feature"]
            ligand_feature = features["ligand_feature"]
            label = features["label"]
            sample_id = features["sample_id"]

            return complex_feature, ligand_feature, label, sample_id

        def process_tfrecord(self, complex_feature, ligand_feature, label, sample_id):
            """Preprocess feature and label parsed from example by parse_example function above.

            Args:
                feature ([type]): [description]
                label ([type]): [description]

            Returns:
                [type]: [description]
            """
            processed_complex_feature = tuple([
                complex_feature[self.complex_feature_names.match(
                    self.regex_template.format(element=element))]  # .astype("float64")
                for element in self.target_elements
            ])
            processed_lgdand_feature = tuple([
                ligand_feature[self.ligand_feature_names.match(
                    self.regex_template.format(element=element))]  # .astype("float64")
                for element in self.target_elements
            ])
            complex_label = label[0]
            ligand_label = label[1]

            return (processed_complex_feature, processed_lgdand_feature),  (complex_label, complex_label, ligand_label)

    def generate_label(self, tfrecords, batch_size=128):
        labels_dataset = self.preprocess(tfrecords).map(
            lambda complex_feature, ligand_feature, labels, sample_id: labels).batch(batch_size)
        labels_array = np.concatenate([np.hstack(
            [col.numpy().reshape([-1, 1]) for col in label]) for label in labels_dataset])
        labels_index = self.generate_index(tfrecords)
        labels = self.shape_preds(labels_array, index=labels_index)
        return labels

    def generate_index(self, tfrecords, batch_size=128):
        feature_names = self.create_feature_name()
        pf = self.PreprocessFunction(
            complex_feature_names=feature_names,
            ligand_feature_names=feature_names,
            target_elements=self.feature_params['target_elements']
        )
        index_dataset = tfrecords.map(pf.parse_example).map(
            lambda complex_feature, ligand_feature, label, sample_id: sample_id).batch(batch_size)
        index = np.vstack([idx for idx in index_dataset]).flatten()
        index = np.vectorize(lambda bytes_obj: bytes_obj.decode())(index)
        return index

    def shape_preds(self, preds, index):
        """function for formatting prediction results into pandas.DataFrame.

        Args:
            preds ([type]): Values of the result dataframe.
            index ([type]): Index of the result dataframe.

        Returns:
            [type]: Formatted prediction result.
        """
        if isinstance(preds, list):
            shaped_preds = pd.DataFrame(np.concatenate(
                preds, axis=1), columns=self.label_colnames, index=index)
        else:
            shaped_preds = pd.DataFrame(
                preds, columns=self.label_colnames, index=index)
        return shaped_preds

    # TFRecord file generation related methods
    @staticmethod
    def _float_feature(value):
        """return float_list from float / double """
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    @staticmethod
    def _bytes_feature(value):
        """return byte_list from string / byte """
        if isinstance(value, type(tf.constant(0))):
            # BytesList won't unpack a string from an EagerTensor.
            value = value.numpy()
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    @staticmethod
    def record2example(cmp_feature, lgd_feature, label, sample_id):
        return tf.train.Example(features=tf.train.Features(feature={
            'complex_feature': ComplexNNplusSubLigandNN._float_feature(cmp_feature),
            'ligand_feature': ComplexNNplusSubLigandNN._float_feature(lgd_feature),
            'label': ComplexNNplusSubLigandNN._float_feature(label),
            'sample_id': ComplexNNplusSubLigandNN._bytes_feature(sample_id)
        }))

    @staticmethod
    def _write_feature_to_tfrecords(complex_feature_files, ligand_feature_files, label_file,
                                    tfr_filename, label_colnames=['pKa_energy', 'ligand_energy']):
        # load features and labels
        feature_cmp = runner.load_feature(
            complex_feature_files).astype('float32')
        feature_lgd = runner.load_feature(
            ligand_feature_files).astype('float32')
        label = runner.load_label(
            label_file, label_colnames=label_colnames).astype('float32')

        # remove not common index
        common_idx = feature_cmp.index.intersection(label.index)
        feature_cmp = feature_cmp.loc[common_idx]
        feature_lgd = feature_lgd.loc[common_idx]

        label = label.loc[common_idx]

        if not all(feature_cmp.index == label.index):
            raise ValueError('feature_cmp and label have different indexes.')
        elif not all(feature_lgd.index == label.index):
            raise ValueError('feature_lgd and label have different indexes.')

        # check whether len of feature is not 0
        if len(feature_cmp.index) == 0 or len(feature_lgd.index) == 0:
            raise ValueError(
                'There are no features to write. Length of feature is 0.')

        # create byte type sample_id
        sample_id = np.vectorize(lambda x: x.encode())(label.index.values)

        # write data to tfr_filename
        n_sample = sample_id.shape[0]
        with tf.io.TFRecordWriter(tfr_filename) as writer:
            for i in tqdm(range(n_sample)):
                ex = ComplexNNplusSubLigandNN.record2example(
                    cmp_feature=feature_cmp.values[i],
                    lgd_feature=feature_lgd.values[i],
                    label=label.values[i],
                    sample_id=sample_id[i]
                )
                writer.write(ex.SerializeToString())
        return None

    @staticmethod
    def generate_feature_tfrecords(complex_feature_files,
                                   ligand_feature_files,
                                   label_file,
                                   tfr_filename,
                                   label_colnames=['pKa_energy', 'ligand_energy']):

        ComplexNNplusSubLigandNN._write_feature_to_tfrecords(
            complex_feature_files,
            ligand_feature_files,
            label_file,
            tfr_filename,
            label_colnames=label_colnames
        )
        print(f"{tfr_filename} was saved...")
        return None

#####################################################################################################
###################     Network Structure classes (use complex feature only)      ###################
#####################################################################################################


class DNN2Input3Output(MyAbstractNetworkStructure):
    """A model that divides the features into radial feature and angular feature and trains radial network and angular network to predict 'pKa', 'pKa_g1', and 'pKa_g2'.

    Args:
        MyAbstractNetworkStructure ([type]): [description]
    """

    def __init__(self, label_colnames=['pKa', 'pKa_g1', 'pKa_g2'], **model_params):
        self.label_colnames = label_colnames
        default_params = dict(
            input_size_radial=512,
            input_size_angular=4096,
            lr=0.0001,
            clipnorm=1,
            l2_norm=0.01
        )
        default_params.update(model_params)
        self.params = default_params
        return None

    def create(self, **kwargs):
        input_size_radial = self.params['input_size_radial']
        input_size_angular = self.params['input_size_angular']
        lr = self.params['lr']
        clipnorm = self.params['clipnorm']
        l2_norm = self.params['l2_norm']

        input_radial = Input(shape=input_size_radial)
        input_angular = Input(shape=input_size_angular)

        x_radial = Dense(256, kernel_regularizer=l2(l2_norm),)(input_radial)
        x_radial = Activation("relu")(x_radial)
        x_radial = BatchNormalization()(x_radial)

        x_angular = Dense(256, kernel_regularizer=l2(l2_norm),)(input_angular)
        x_angular = Activation("relu")(x_angular)
        x_angular = BatchNormalization()(x_angular)

        x_combined = concatenate([x_radial, x_angular])

        x = Dense(256, kernel_regularizer=l2(l2_norm),)(x_combined)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)

        x = Dense(256, kernel_regularizer=l2(l2_norm),)(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)

        x = Dense(256, kernel_regularizer=l2(l2_norm),)(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)

        x = Dense(256, kernel_regularizer=l2(l2_norm),)(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)

        x_g = Dense(128, kernel_regularizer=l2(l2_norm),)(x)
        x_g = Activation("relu")(x_g)
        x_g = BatchNormalization()(x_g)
        output_g = Dense(1, name="output_g")(x_g)

        x_g1 = Dense(128, kernel_regularizer=l2(l2_norm),)(x)
        x_g1 = Activation("relu")(x_g1)
        x_g1 = BatchNormalization()(x_g1)
        output_g1 = Dense(1, name="output_g1")(x_g1)

        x_g2 = Dense(128, kernel_regularizer=l2(l2_norm),)(x)
        x_g2 = Activation("relu")(x_g2)
        x_g2 = BatchNormalization()(x_g2)
        output_g2 = Dense(1, name="output_g2")(x_g2)

        # ValueError: Gradient clipping in the optimizer
        # (by setting clipnorm or clipvalue)
        # setting clipnorm or clipvalue is currently unsupported when using a distribution strategy.
        optimizer = SGD(
            lr=lr,
            momentum=0.9,
            decay=1e-6,
            # clipnorm=clipnorm
        )

        network = tf.keras.models.Model(
            inputs=[input_radial, input_angular],
            outputs=[output_g, output_g1, output_g2]
        )

        network.compile(
            loss={'output_g': 'mse', 'output_g1': 'mse', 'output_g2': 'mse'},
            loss_weights={'output_g': 1, 'output_g1': 1, 'output_g2': 1},
            optimizer=optimizer,
            metrics={
                'output_g': ['mae', 'mse'],
                'output_g1': ['mae', 'mse'],
                'output_g2': ['mae', 'mse']
            }
        )

        return network

    def preprocess(self, X, y=None, **kwargs):
        X_radial = X.iloc[:, :self.params['input_size_radial']].astype(
            "float64")
        X_angular = X.iloc[:, self.params['input_size_radial']:].astype(
            "float64")
        _X = (X_radial, X_angular)
        if y is None:
            return _X
        else:
            _y = y.copy().values.astype("float64")
            _y = _y[:, 0], _y[:, 1], _y[:, 2]
            return _X, _y

    def shape_preds(self, preds, index):
        return pd.DataFrame(np.concatenate(preds, axis=1), columns=self.label_colnames, index=index)

    def compile(self, network):
        optimizer = SGD(
            lr=self.params['lr'],
            momentum=0.9,
            decay=1e-6,
            # clipnorm=clipnorm
        )

        network.compile(
            loss={'output_g': 'mse', 'output_g1': 'mse', 'output_g2': 'mse'},
            loss_weights={'output_g': 1, 'output_g1': 1, 'output_g2': 1},
            optimizer=optimizer,
            metrics={
                'output_g': ['mae', 'mse'],
                'output_g1': ['mae', 'mse'],
                'output_g2': ['mae', 'mse']
            }
        )
        return network

    @classmethod
    def calculate_metrics(cls, y_true, y_preds):
        score = {}
        mse = {f"{column_name} mse": mean_squared_error(y_true=y_t, y_pred=y_p) for (
            column_name, y_t), (_, y_p) in zip(y_true.iteritems(), y_preds.iteritems())}
        r2 = {f"{column_name} r2": r2_score(y_true=y_t, y_pred=y_p) for (
            column_name, y_t), (_, y_p) in zip(y_true.iteritems(), y_preds.iteritems())}
        score.update(mse)
        score.update(r2)
        return score


#####################################################################################################
###################           Network Structure classes (prototypes)              ###################
#####################################################################################################

class ElementwiseDNN_prev(MyAbstractNetworkStructure):
    def __init__(self, label_colnames=['pKa'], **model_params):
        self.label_colnames = label_colnames

        default_params = dict(
            input_size_per_element=576,  # 4096/8 + 512/8,
            target_elements=['H', 'C', 'N', 'O', 'P', 'S', 'Cl', 'DU'],
            n_target_elements=8,
            lr=0.0001,
            clipnorm=1
        )
        default_params.update(model_params)
        self.model_params = default_params
        return None

    def create_dnn_for_one_element(self):
        input_size = self.model_params['input_size_per_element']
        lr = self.model_params['lr']
        clipnorm = self.model_params['clipnorm']
        l2_norm = self.model_params['l2_norm']

        input = Input(shape=input_size)

        x = Dense(128, kernel_regularizer=l2(l2_norm),)(input)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)

        x = Dense(128, kernel_regularizer=l2(l2_norm),)(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)

        x = Dense(64, kernel_regularizer=l2(l2_norm),)(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)

        output = Dense(1)(x)

        network = tf.keras.models.Model(
            inputs=[input],
            outputs=[output]
        )
        return network

    def create(self):
        n_target_elements = self.feature_params['n_target_elements']
        input_size_per_element = self.model_params['input_size_per_element']
        lr = self.model_params['lr']
        clipnorm = self.model_params['clipnorm']

        inputs = [Input(shape=input_size_per_element)
                  for _ in range(n_target_elements)]
        elementwise_models = [self.create_dnn_for_one_element()
                              for _ in range(n_target_elements)]

        outputs = [model(input)
                   for model, input in zip(elementwise_models, inputs)]
        output = tf.keras.layers.Add(name='output_sum')(outputs)

        network = tf.keras.models.Model(
            inputs=[inputs],
            outputs=[output]
        )

        optimizer = SGD(
            lr=lr,
            momentum=0.9,
            decay=1e-6,
            # clipnorm=clipnorm
        )

        network.compile(
            loss={'output_sum': 'mse'},
            optimizer=optimizer,
            metrics={
                'output_sum': ['mae', 'mse'],
            }
        )

        return network

    # previous version
    # def preprocess(self, X, y=None, **network_cls_kwargs):
    #     target_elements = self.feature_params['target_elements']
    #     _X = [X.loc[:,X.columns.str.startswith(element + '_')].astype("float64") for element in target_elements]
    #     if y is None:
    #         return _X
    #     else:
    #         _y = y.values[:,0].astype("float64").copy()
    #         return _X, _y

    def preprocess(self, X, y=None):
        target_elements = self.feature_params['target_elements']
        _X = [X.loc[:, X.columns.str.match(
            f'(^\w?\w?_{element}_\w?\w?_\d+_\d+$)|(^{element}_\w?\w?_\d+$)')
        ].astype("float64")
            for element in target_elements]
        if y is None:
            return _X
        else:
            _y = y.values[:, 0].astype("float64").copy()
            return _X, _y

    @classmethod
    def calculate_metrics(cls, y_true, y_preds):
        mse = mean_squared_error(y_true=y_true, y_pred=y_preds)
        r2 = r2_score(y_true=y_true, y_pred=y_preds)
        return {'mse': mse, 'r2': r2}

    def shape_preds(self, preds, index):
        return pd.DataFrame(preds, columns=self.label_colnames, index=index)

    def compile(self, network):
        optimizer = SGD(
            lr=self.model_params['lr'],
            momentum=0.9,
            decay=1e-6,
            # clipnorm=clipnorm
        )

        network.compile(
            loss={'output_sum': 'mse'},
            optimizer=optimizer,
            metrics={
                'output_sum': ['mae', 'mse'],
            }
        )
        return network
