# import os
# import sys
import tempfile
import itertools
# import logging
# import warnings
# import multiprocessing
# from collections import OrderedDict
# from collections import defaultdict
# from collections import Counter

import numpy as np
import pandas as pd
# import dask.dataframe as ddf
# from dask.diagnostics import ProgressBar
import mdtraj as mt
from biopandas.pdb import PandasPdb
from numba import vectorize, float64


class LigandProteinComplex(object):
    """Extract feature of protein-ligand interaction from pdb file.

     The series of functions to extract feature of protein-ligand interaction from pdb file.
     This class is mainly composed of three functions: parsePDB(), set_shell_structure() and generate_feature_withRBF().
     Please see Examples below.

     args:
        pdb_file (str)): The input pdb file name.
        lig_code (str): The ligand residue name in the input pdb file.

    Attributes:
        pdb (mdtraj.Trajectory): The mdtraj.trajectory object containing the pdb.
        receptor_indices (np.ndarray): The ligand residue name in the input pdb file.
        ligand_indices (np.ndarray): The ligand (protein) atom indices in mdtraj.Trajectory
    
    Examples:
        >>> cplx = LigandProteinComplex('path/to/pdb/file.pdb', lig_code = "LGD")
        >>> cplx.parsePDB(source='onionnet')
        >>> radial_feature_name, radial_feature_value = cplx.generate_radial_feature()
        >>> angular_feature_name, angular_feature_value = cplx.generate_angular_feature()


    """
    def __init__(self, pdb_file, lig_code="LGD", 
                target_elements=["H", "C", "N", "O", "P", "S", "Cl", "DU"],
                distance_threshold_radial=4, distance_threshold_angular=4,
                Rs_list_radial=[0.5, 1.17, 1.83, 2.5, 3.17, 3.83, 4.5, 5.17],
                Rs_list_angular=[0.5, 1.17, 1.83, 2.5, 3.17, 3.83, 4.5, 5.17],
                theta_list=[0, 0.785, 1.57, 2.355, 3.14, 3.925, 4.71, 5.495]):
        
        self.pdb_file = pdb_file
        self.lig_code = lig_code

        self.receptor_indices = np.array([])
        self.ligand_indices = np.array([])

        self.rec_ele = np.array([])
        self.lig_ele = np.array([])

        self.is_pdb_parsed_ = False
        # self.distance_computed_ = False

        # self.distance_matrix_ = np.array([])
        # self.counts_ = np.array([])
        
        # self.angle_shell_structure = {}
        # self.is_angle_shell_structure_set_ = False
        # self.angle_feature = None
        
        # self.shell_structure = {}
        # self.is_shell_structure_set_ = False
        # self.feature = None

        self.target_elements = target_elements
        self.distance_threshold_radial = distance_threshold_radial
        self.distance_threshold_angular = distance_threshold_angular
        self.Rs_list_radial = Rs_list_radial
        self.Rs_list_angular = Rs_list_angular
        self.theta_list = theta_list
        
        return None
        

    def parsePDB(self, is_pdbformat_converted=False, source='onionnet'):
        """parse PDB file and set self.receptor_indices and self.ligand_indices.

         parse asedock result pdb file. 

        Args:
            is_pdbformat_converted (bool): Whether input pdb file format is converted. 
            (True is preprocessed, False means that the pdb files need to convert its format.)
            source (str): Source of PDB file. 'onionnet' or 'pdb_archive'. 
                'onionnet' means  the pdbfile which is preprocessed with same workflow as onionnet.
                'pdb_archive' means the pdbfile which is downloaded from pdb archive and not preprocessed. 

        Returns:
           self: An instance with self.pdb attribute.

        Examples:
            >>> cplx = LigandProteinComplex('path/to/pdb/file.pdb', lig_code = "LGD")
            >>> cplx.parsePDB(is_pdbformat_converted=False, source='onionnet')

        Note:
            This functions is optimized to asedock output pdbfile. Other pdb file formats are not tested.
            Tempfile is generated when is_pdbformat_converted is False. (immediately removed)
            Tempfile path is Temp/tmp{randomstring}.pdb

        """
        if is_pdbformat_converted:            
            self.pdb = mt.load(self.pdb_file)
            
        elif source == 'onionnet':
            self.pdb = self.convert_pdbfileformat_from_onionnet(self.pdb_file)
            
        elif source == 'pdb_archive':
            self.pdb = self.convert_pdbfileformat_from_archive(self.pdb_file)
        else:
            raise ValueError("source argument must be either 'onionnet' or 'pdb_archive'" )

        top = self.pdb.topology

        self.receptor_indices = top.select("not resname " + self.lig_code) # top.select("protein")
        self.ligand_indices = top.select("resname " + self.lig_code)

        table, _ = top.to_dataframe()

        self.rec_ele = table['element'][self.receptor_indices]
        self.lig_ele = table['element'][self.ligand_indices]

        self.is_pdb_parsed_ = True

        return self
    
    def convert_pdbfileformat_from_onionnet(self, pdb_file):
        ppdb = PandasPdb()
        ppdb.read_pdb(pdb_file)
        ppdb.df['ATOM']=pd.concat([ppdb.df['ATOM'].query(f"residue_name!='{self.lig_code}'"), 
                                   ppdb.df['ATOM'].query(f"residue_name=='{self.lig_code}'")])

        # WARNING: Tempfile will be generated. (Immediately removed)
        with tempfile.NamedTemporaryFile(suffix='.pdb') as temp:
            ppdb.to_pdb(path=temp.name, 
                        records=['ATOM'], 
                        gz=False, 
                        append_newline=True)
            converted_pdb = mt.load(temp.name)
        
        return  converted_pdb # mdtraj.Trajectory object
    
    def convert_pdbfileformat_from_archive(self, pdb_file):
        ppdb = PandasPdb()
        ppdb.read_pdb(pdb_file)

        ligand_and_others = ppdb.df['HETATM'].query("residue_name!='HOH'")
        ligand_residue_name = ligand_and_others.residue_name.value_counts().idxmax()
        ligand_and_others.loc[:, 'residue_name'].replace(ligand_residue_name, self.lig_code, inplace=True)
        ppdb.df['HETATM'] = ligand_and_others
        
        with tempfile.NamedTemporaryFile(suffix='.pdb') as temp:
            ppdb.to_pdb(path=temp.name, 
                        records=['ATOM', 'HETATM'], 
                        gz=False, 
                        append_newline=True)

            converted_pdb = mt.load(temp.name)
        return converted_pdb
        
    @staticmethod
    def is_angle_pi(li_xyz, rj_xyz, rk_xyz):
        return all(~((rj_xyz < li_xyz) ^ (li_xyz < rk_xyz)))

    def check_zero_or_pi(self, index_triplet):
        R_j, L_i, R_k = index_triplet
        li_xyz = self.pdb.xyz[0][L_i]
        rj_xyz = self.pdb.xyz[0][R_j]
        rk_xyz = self.pdb.xyz[0][R_k]
        if LigandProteinComplex.is_angle_pi(li_xyz, rj_xyz, rk_xyz):
            return np.pi
        return 0
    
    # def _get_elementtype(self, e):
    #     """Convert NOT target elements to DU 

    #     Convert element to DU if the element are not listed on self.target_elements,
    #     else return itself.

    #     Args:
    #         e (str): element type. e.g) "H" "C" "N"

    #     Returns:
    #        str: element type.

    #     """
    #     if e in self.shell_structure["self.target_elements"]:
    #         return e
    #     else:
    #         return "DU"   

    # @staticmethod
    def _get_elementtype(self, e):
        """Convert NOT target elements to DU 

        Convert element to DU if the element are not listed on self.target_elements,
        else return itself.

        Args:
            e (str): element type. e.g) "H" "C" "N"

        Returns:
            str: element type.

        """
        # self.target_elements = ["H", "C", "N", "O", "P", "S", "Cl", "DU"]
        if e in self.target_elements:
            return e
        else:
            return "DU"

    @staticmethod
    def radial_symmetry_function(Rij, Rs, Rc=6, eta=4):
        # Rij_ang = Rij * 10
        fc = 0.5 * np.cos(np.pi * Rij / Rc) + 0.5
        return np.exp(- eta * (Rij - Rs) ** 2) * fc
        
    @staticmethod
    @vectorize([float64(float64, float64, float64, float64)])
    def radial_symmetry_function_vectorized(Rij, Rs, Rc=6, eta=4):
        fc = 0.5 * np.cos(np.pi * Rij / Rc) + 0.5
        return np.exp(- eta * (Rij - Rs) ** 2) * fc

    def generate_radial_feature(self):

        target_elements_product = [f"{e_l}_{e_r}" for e_l, e_r in itertools.product(self.target_elements, repeat=2)]
        elements_product2elements_product_id = dict(zip(
            target_elements_product,
            range(len(target_elements_product))
        ))

        atomidx2element = dict(zip(
            np.concatenate([self.rec_ele.index, self.lig_ele.index]),
            np.concatenate([self.rec_ele.values, self.lig_ele.values])
        ))

        Rs2index = dict(zip(self.Rs_list_radial, range(len(self.Rs_list_radial))))
        Rs_index = list(map(Rs2index.get, self.Rs_list_radial))
        feature_name = [f"{e_l}_{e_r}_{rs}" for e_l, e_r in itertools.product(self.target_elements, repeat=2) for rs in Rs_index]

        lig_rec_pair = np.fromiter(itertools.chain(*itertools.product(self.ligand_indices, self.receptor_indices)),dtype=int).reshape(-1,2)

        distance_matrix = mt.compute_distances(self.pdb, lig_rec_pair)[0]
        # convert unit from nm to angstrom
        distance_matrix = distance_matrix * 10 
        mask = distance_matrix < self.distance_threshold_radial

        # check whether there are atoms inside the borde
        # if not any(mask):
        #     return feature_name, np.zeros(len(feature_name), dtype='float64')
        inside_border_pair = lig_rec_pair[mask]
        distance_array = distance_matrix[mask]

        # if inside_border_pair.size == 0:
        #     feature_value = np.zeros(len(self.target_elements)**2 * len(self.Rs_list_radial))
        #     return feature_name, feature_value

        inside_border_pair_element = np.vectorize(atomidx2element.get, otypes=['<U2'])(inside_border_pair)
        inside_border_pair_element = np.frompyfunc(self._get_elementtype, 1, 1)(inside_border_pair_element)
        # apply_along_axis returns wrong dtype (<U3). 
        # If you go over three letters, you lose the excess. (Cl_H -> Cl_ (H was lost))
        # inside_border_pair_element = np.apply_along_axis(lambda x: '_'.join(x), 1, inside_border_pair_element)
        inside_border_pair_element = np.array(['_'.join(pair) for pair in inside_border_pair_element])
        inside_border_pair_element_id = np.vectorize(
            elements_product2elements_product_id.get, otypes=['int64']
        )(inside_border_pair_element)

        distance_mesh, Rs_mesh = np.meshgrid(distance_array, self.Rs_list_radial)

        # radial_symmetry_function_ufunc = np.frompyfunc(LigandProteinComplex.radial_symmetry_function, 2, 1)
        # pair_x_rs = radial_symmetry_function_ufunc(distance_mesh, Rs_mesh).astype(float).T

        # cannot write args of vectrize function as (distance_mesh, Rs_mesh, Rc=Rc, eta=4)
        pair_x_rs = LigandProteinComplex.radial_symmetry_function_vectorized(
            distance_mesh, Rs_mesh, self.distance_threshold_radial, 4).T # 6, 4

        element_x_rs = np.apply_along_axis(
            lambda x: np.bincount(inside_border_pair_element_id, weights=x, minlength=len(target_elements_product)), 
            axis=0, 
            arr=pair_x_rs
        )
        
        # feature_value = np.append(element_x_rs.reshape(-1), np.zeros(len(target_elements_product) * len(self.Rs_list_radial) - len(element_x_rs.reshape(-1))))
        feature_value = element_x_rs.reshape(-1)

        return feature_name, feature_value

    @staticmethod
    def angular_symmetry_function(Rij, Rik, theta_ijk, theta, Rs = 1, Rc=6, zeta=8, eta=4):
        # Rij = Rij_nm * 10
        # Rik = Rik_nm * 10
        fc_Rij = 0.5 * np.cos(np.pi * Rij / Rc) + 0.5
        fc_Rik = 0.5 * np.cos(np.pi * Rik / Rc) + 0.5

        return 2**(1 - zeta) * (1 + np.cos(theta_ijk - theta))**zeta * \
            np.exp(- eta * ((Rij + Rik) / 2 - Rs) ** 2) * fc_Rij * fc_Rik

    @staticmethod
    @vectorize([float64(float64, float64, float64, float64, float64, float64, float64, float64)])
    def angular_symmetry_function_vectorized(Rij, Rik, theta_ijk, theta, Rs=1, Rc=6, zeta=8, eta=4):
        fc_Rij = 0.5 * np.cos(np.pi * Rij / Rc) + 0.5
        fc_Rik = 0.5 * np.cos(np.pi * Rik / Rc) + 0.5
        return 2**(1 - zeta) * (1 + np.cos(theta_ijk - theta))**zeta * \
            np.exp(- eta * ((Rij + Rik) / 2 - Rs) ** 2) * fc_Rij * fc_Rik

    def generate_angular_feature(self):

        # target_elements_triplet = [f"{e_j}_{e_i}_{e_k}" for e_j, e_i, e_k in itertools.product(self.target_elements, repeat=3)]
        target_elements_triplet =  [f"{e_j}_{e_i}_{e_k}" for (e_i, (e_j, e_k)) in 
            list(itertools.product(
                self.target_elements,
                list(itertools.combinations_with_replacement(self.target_elements, 2))
            ))
        ]

        elements_triplet2elements_triplet_id = dict(zip(
            target_elements_triplet,
            range(len(target_elements_triplet))
        ))

        atomidx2element = dict(zip(
            np.concatenate([self.rec_ele.index, self.lig_ele.index]),
            np.concatenate([self.rec_ele.values, self.lig_ele.values])
        ))

        # thetas2index = dict(zip(self.theta_list, range(len(self.theta_list))))
        thetas_index = list(range(len(self.theta_list)))
        Rs_index = list(range(len(self.Rs_list_angular)))

        # feature_name = [f"{e_j}_{e_i}_{e_k}_{theta}_{rs}" 
        #     for e_j, e_i, e_k in itertools.product(self.target_elements, repeat=3) 
        #         for theta in thetas_index
        #             for rs in Rs_index]

        feature_name = [f"{element_triplet}_{theta}_{rs}" 
            for element_triplet in target_elements_triplet
                for theta in thetas_index
                    for rs in Rs_index]


        lig_rec_pair = np.fromiter(
            itertools.chain(*itertools.product(self.ligand_indices, self.receptor_indices)), dtype=int
        ).reshape(-1,2)

        distance_matrix = mt.compute_distances(self.pdb, lig_rec_pair)
        # convert unit from nm to angstrom
        distance_matrix = distance_matrix * 10 
        mask = distance_matrix[0] < self.distance_threshold_angular
        inside_border_pair = lig_rec_pair[mask]
        # distance_array = distance_matrix[0][mask]

        target_ligands = inside_border_pair[:, 0]
        target_receptors = np.split(inside_border_pair[:, 1], 
            np.cumsum(np.unique(inside_border_pair[:, 0], return_counts=True)[1])[:-1])
        target_triplets = np.array([(r_j, l_i, r_k) 
            for l_i, target_receptor in zip(np.unique(target_ligands), target_receptors) 
                for r_j, r_k  in itertools.combinations(target_receptor, 2)
        ])
        # check whether target_triplets is empty.
        # if empty, return zeros
        if target_triplets.size == 0:
            feature_value = np.zeros(len(target_elements_triplet) * len(self.theta_list) * len(self.Rs_list_angular))
            return feature_name, feature_value

        # convert atom triplet index to element triplet id 
        target_triplet_element = np.vectorize(atomidx2element.get, otypes=['<U2'])(target_triplets)
        target_triplet_element = np.frompyfunc(self._get_elementtype, 1, 1)(target_triplet_element)
        
        # sort r_j, r_k by elementid
        element2elementid = dict(zip(self.target_elements, range(len(self.target_elements))))
        elementid2element = dict(zip(range(len(self.target_elements)), self.target_elements))

        target_receptors_element_id = np.vectorize(element2elementid.get, otypes=['int'])(target_triplet_element[:,[0,2]])
        target_receptors_element_id_sorted = np.sort(target_receptors_element_id)
        target_receptors_element = np.vectorize(elementid2element.get, otypes=['<U2'])(target_receptors_element_id_sorted)
        target_triplet_element_sorted = np.block([
            target_receptors_element[:, 0].reshape([-1, 1]), 
            target_triplet_element[:, 1].reshape([-1, 1]), 
            target_receptors_element[:, 1].reshape([-1, 1])
        ])
        # apply_along_axis returns wrong dtype (<U5). 
        # If you go over five letters, you lose the excess. (N_Cl_H -> 'N_Cl_' (H was lost))
        target_triplet_element_str = np.array(['_'.join(triplet) for triplet in target_triplet_element_sorted])
        target_triplet_element_id = np.vectorize(elements_triplet2elements_triplet_id.get, otypes=['int64'])(target_triplet_element_str)

        theta_jik = mt.compute_angles(self.pdb, angle_indices=target_triplets)[0]
        theta_jik = np.array([
            angle if not np.isnan(angle) else self.check_zero_or_pi(target_triplet)  
                for angle, target_triplet in zip(theta_jik, target_triplets)
        ]).reshape(-1,1)

        r_ij = mt.compute_distances(self.pdb, target_triplets[:,[1,0]]).reshape(-1,1)
        r_ik = mt.compute_distances(self.pdb, target_triplets[:,[1,2]]).reshape(-1,1)
        # convert unit from nm to angstrom
        r_ij = r_ij * 10
        r_ik = r_ik * 10
        distance_and_angle = np.concatenate([r_ij, r_ik, theta_jik], 1)

        # Add Rs list to mesh dimension
        r_ij_mesh = np.tile(r_ij, len(self.theta_list) * len(self.Rs_list_angular))\
            .reshape(-1, len(self.theta_list), len(self.Rs_list_angular), 1)
        r_ik_mesh = np.tile(r_ik, len(self.theta_list) * len(self.Rs_list_angular))\
            .reshape(-1, len(self.theta_list), len(self.Rs_list_angular), 1)
        theta_jik_mesh = np.tile(theta_jik, len(self.theta_list) * len(self.Rs_list_angular))\
            .reshape(-1, len(self.theta_list), len(self.Rs_list_angular), 1)

        thetas_mesh = np.tile(self.theta_list, (distance_and_angle.shape[0], len(self.Rs_list_angular)))\
            .reshape(-1, len(self.Rs_list_angular), len(self.theta_list), 1)\
            .transpose(0, 2, 1, 3)
        Rs_mesh = np.tile(self.Rs_list_angular, (distance_and_angle.shape[0], len(self.theta_list)))\
            .reshape(-1, len(self.theta_list), len(self.Rs_list_angular), 1)

        
        # x3 speed up by using vectorized angular_symmetry_function
        triplet_x_thetas_x_Rs = LigandProteinComplex.angular_symmetry_function_vectorized(
            r_ij_mesh, r_ik_mesh, theta_jik_mesh, thetas_mesh, Rs_mesh, 
            self.distance_threshold_angular, 8, 4) # 6, 8, 4

        element_x_thetas_x_Rs = np.apply_along_axis(
            lambda x: np.bincount(target_triplet_element_id, weights=x, minlength=len(target_elements_triplet)), 
            axis=0, arr=triplet_x_thetas_x_Rs
        )
        feature_value = element_x_thetas_x_Rs.reshape(-1)

        return feature_name, feature_value

    def generate_radial_feature_onlyligand(self):

        target_elements_product = [f"{e_l}_{e_r}" 
            for e_l, e_r in itertools.product(self.target_elements, repeat=2)
        ]
        elements_product2elements_product_id = dict(zip(
            target_elements_product,
            range(len(target_elements_product))
        ))

        atomidx2element = dict(zip(
            np.concatenate([self.lig_ele.index, self.lig_ele.index]),
            np.concatenate([self.lig_ele.values, self.lig_ele.values])
        ))

        Rs2index = dict(zip(self.Rs_list_radial, range(len(self.Rs_list_radial))))
        Rs_index = list(map(Rs2index.get, self.Rs_list_radial))
        feature_name = [f"{e_l}_{e_r}_{rs}" for e_l, e_r in itertools.product(self.target_elements, repeat=2) for rs in Rs_index]

        lig_lig_pair = np.fromiter(itertools.chain(*itertools.permutations(self.ligand_indices, 2)),dtype=int).reshape(-1,2)

        distance_matrix = mt.compute_distances(self.pdb, lig_lig_pair)[0]
        # convert unit from nm to angstrom
        distance_matrix = distance_matrix * 10 

        lig_lig_pair_element = np.vectorize(atomidx2element.get)(lig_lig_pair)
        lig_lig_pair_element = np.frompyfunc(self._get_elementtype, 1, 1)(lig_lig_pair_element)
        # apply_along_axis returns wrong dtype (<U3). 
        # If you go over three letters, you lose the excess. (Cl_H -> Cl_ (H was lost))
        lig_lig_pair_element = np.array(['_'.join(pair) for pair in lig_lig_pair_element])
        lig_lig_pair_element_id = np.vectorize(elements_product2elements_product_id.get)(lig_lig_pair_element)

        distance_mesh, Rs_mesh = np.meshgrid(distance_matrix, self.Rs_list_radial)
        pair_x_rs = LigandProteinComplex.radial_symmetry_function_vectorized(distance_mesh, Rs_mesh, 6, 4).T

        element_x_rs = np.apply_along_axis(
            lambda x: np.bincount(lig_lig_pair_element_id, weights=x, minlength=len(target_elements_product)), 
            axis=0, 
            arr=pair_x_rs
        )
        feature_value = element_x_rs.reshape(-1)

        return feature_name, feature_value

    def generate_angular_feature_onlyligand(self):

        target_elements_triplet = [f"{e_j}_{e_i}_{e_k}" for e_j, e_i, e_k in itertools.product(self.target_elements, repeat=3)]
        elements_triplet2elements_triplet_id = dict(zip(
            target_elements_triplet,
            range(len(target_elements_triplet))
        ))

        atomidx2element = dict(zip(self.lig_ele.index, self.lig_ele.values))

        # thetas2index = dict(zip(self.theta_list, range(len(self.theta_list))))
        thetas_index = list(range(len(self.theta_list)))
        # feature_name = [f"{e_j}_{e_i}_{e_k}_{theta}" for e_j, e_i, e_k in itertools.product(self.target_elements, repeat=3) for theta in thetas_index]

        Rs_index = list(range(len(self.Rs_list_angular)))
        feature_name = [f"{e_j}_{e_i}_{e_k}_{theta}_{rs}" 
            for e_j, e_i, e_k in itertools.product(self.target_elements, repeat=3) 
                for theta in thetas_index
                    for rs in Rs_index]

        target_other_ligand = np.fromiter(itertools.chain(*itertools.combinations(self.ligand_indices,2)), dtype=int).reshape(-1,2)
        target_triplets = np.array([(r_j, l_i, r_k) 
            for l_i, (r_j, r_k) in itertools.product(self.ligand_indices, target_other_ligand)
                if not (l_i == r_j) and not(l_i == r_k)
        ])

        # convert atom triplet index to element triplet id 
        target_triplet_element = np.vectorize(atomidx2element.get, otypes=['<U2'])(target_triplets)
        target_triplet_element = np.frompyfunc(self._get_elementtype, 1, 1)(target_triplet_element)
        # apply_along_axis returns wrong dtype (<U5). 
        # If you go over five letters, you lose the excess. (N_Cl_H -> 'N_Cl_' (H was lost))
        target_triplet_element = np.array(['_'.join(triplet) for triplet in target_triplet_element])
        target_triplet_element_id = np.vectorize(elements_triplet2elements_triplet_id.get, otypes=['int64'])(target_triplet_element)

        # check whether target_triplets is empty.
        # if empty, return zeros
        if target_triplets.size == 0:
            feature_value = np.zeros(len(target_elements_triplet) * len(self.theta_list) * len(self.Rs_list_angular))
            return feature_name, feature_value

        theta_jik = mt.compute_angles(self.pdb, angle_indices=target_triplets)[0]
        theta_jik = np.array([
            angle if not np.isnan(angle) else self.check_zero_or_pi(target_triplet)  
                for angle, target_triplet in zip(theta_jik, target_triplets)
        ]).reshape(-1,1)


        r_ij = mt.compute_distances(self.pdb, target_triplets[:,[1,0]]).reshape(-1,1)
        r_ik = mt.compute_distances(self.pdb, target_triplets[:,[1,2]]).reshape(-1,1)
        # convert unit from nm to angstrom
        r_ij = r_ij * 10
        r_ik = r_ik * 10
        distance_and_angle = np.concatenate([r_ij, r_ik, theta_jik], 1)

        # Add Rs list to mesh dimension
        r_ij_mesh = np.tile(r_ij, len(self.theta_list) * len(self.Rs_list_angular)).reshape(-1, len(self.theta_list), len(self.Rs_list_angular), 1)
        r_ik_mesh = np.tile(r_ik, len(self.theta_list) * len(self.Rs_list_angular)).reshape(-1, len(self.theta_list), len(self.Rs_list_angular), 1)
        theta_jik_mesh = np.tile(theta_jik, len(self.theta_list) * len(self.Rs_list_angular)).reshape(-1, len(self.theta_list), len(self.Rs_list_angular), 1)

        thetas_mesh = np.tile(self.theta_list, (distance_and_angle.shape[0], len(self.Rs_list_angular)))\
            .reshape(-1, len(self.Rs_list_angular), len(self.theta_list), 1)\
            .transpose(0, 2, 1, 3)
        Rs_mesh = np.tile(self.Rs_list_angular, (distance_and_angle.shape[0], len(self.theta_list)))\
            .reshape(-1, len(self.theta_list), len(self.Rs_list_angular), 1)

        
        # x3 speed up by using vectorized angular_symmetry_function
        triplet_x_thetas_x_Rs = LigandProteinComplex.angular_symmetry_function_vectorized(r_ij_mesh, r_ik_mesh, theta_jik_mesh, thetas_mesh, Rs_mesh, 6, 8, 4)

        element_x_thetas_x_Rs = np.apply_along_axis(
            lambda x: np.bincount(target_triplet_element_id, weights=x, minlength=len(target_elements_triplet)), 
            axis=0, arr=triplet_x_thetas_x_Rs
        )
        feature_value = element_x_thetas_x_Rs.reshape(-1)

        return feature_name, feature_value


    def __generate_angular_feature(self):

        target_elements_triplet = [f"{e_j}_{e_i}_{e_k}" for e_j, e_i, e_k in itertools.product(self.target_elements, repeat=3)]
        elements_triplet2elements_triplet_id = dict(zip(
            target_elements_triplet,
            range(len(target_elements_triplet))
        ))

        atomidx2element = dict(zip(
            np.concatenate([self.rec_ele.index, self.lig_ele.index]),
            np.concatenate([self.rec_ele.values, self.lig_ele.values])
        ))

        # thetas2index = dict(zip(self.theta_list, range(len(self.theta_list))))
        thetas_index = list(range(len(self.theta_list)))
        # feature_name = [f"{e_j}_{e_i}_{e_k}_{theta}" for e_j, e_i, e_k in itertools.product(self.target_elements, repeat=3) for theta in thetas_index]

        Rs_index = list(range(len(self.Rs_list_angular)))
        feature_name = [f"{e_j}_{e_i}_{e_k}_{theta}_{rs}" 
            for e_j, e_i, e_k in itertools.product(self.target_elements, repeat=3) 
                for theta in thetas_index
                    for rs in Rs_index]


        lig_rec_pair = np.fromiter(itertools.chain(*itertools.product(self.ligand_indices, self.receptor_indices)),dtype=int).reshape(-1,2)

        distance_matrix = mt.compute_distances(self.pdb, lig_rec_pair)
        # convert unit from nm to angstrom
        distance_matrix = distance_matrix * 10 
        mask = distance_matrix[0] < self.distance_threshold_angular
        inside_border_pair = lig_rec_pair[mask]
        # distance_array = distance_matrix[0][mask]

        target_ligands = inside_border_pair[:, 0]
        target_receptors = np.split(inside_border_pair[:, 1], 
            np.cumsum(np.unique(inside_border_pair[:, 0], return_counts=True)[1])[:-1])
        target_triplets = np.array([(r_j, l_i, r_k) 
            for l_i, target_receptor in zip(target_ligands, target_receptors) 
                for r_j, r_k  in itertools.combinations(target_receptor, 2)
        ])

        # convert atom triplet index to element triplet id 
        target_triplet_element = np.vectorize(atomidx2element.get, otypes=['<U2'])(target_triplets)
        target_triplet_element = np.frompyfunc(self._get_elementtype, 1, 1)(target_triplet_element)
        # apply_along_axis returns wrong dtype (<U5). 
        # If you go over five letters, you lose the excess. (N_Cl_H -> 'N_Cl_' (H was lost))
        target_triplet_element = np.array(['_'.join(triplet) for triplet in target_triplet_element])
        target_triplet_element_id = np.vectorize(elements_triplet2elements_triplet_id.get, otypes=['int64'])(target_triplet_element)

        # check whether target_triplets is empty.
        # if empty, return zeros
        if target_triplets.size == 0:
            feature_value = np.zeros(len(target_elements_triplet) * len(self.theta_list) * len(self.Rs_list_angular))
            return feature_name, feature_value

        theta_jik = mt.compute_angles(self.pdb, angle_indices=target_triplets)[0]
        theta_jik = np.array([
            angle if not np.isnan(angle) else self.check_zero_or_pi(target_triplet)  
                for angle, target_triplet in zip(theta_jik, target_triplets)
        ]).reshape(-1,1)

        r_ij = mt.compute_distances(self.pdb, target_triplets[:,[1,0]]).reshape(-1,1)
        r_ik = mt.compute_distances(self.pdb, target_triplets[:,[1,2]]).reshape(-1,1)
        # convert unit from nm to angstrom
        r_ij = r_ij * 10
        r_ik = r_ik * 10
        distance_and_angle = np.concatenate([r_ij, r_ik, theta_jik], 1)

        # Add Rs list to mesh dimension
        r_ij_mesh = np.tile(r_ij, len(self.theta_list) * len(self.Rs_list_angular)).reshape(-1, len(self.theta_list), len(self.Rs_list_angular), 1)
        r_ik_mesh = np.tile(r_ik, len(self.theta_list) * len(self.Rs_list_angular)).reshape(-1, len(self.theta_list), len(self.Rs_list_angular), 1)
        theta_jik_mesh = np.tile(theta_jik, len(self.theta_list) * len(self.Rs_list_angular)).reshape(-1, len(self.theta_list), len(self.Rs_list_angular), 1)

        thetas_mesh = np.tile(self.theta_list, (distance_and_angle.shape[0], len(self.Rs_list_angular)))\
            .reshape(-1, len(self.Rs_list_angular), len(self.theta_list), 1)\
            .transpose(0, 2, 1, 3)
        Rs_mesh = np.tile(self.Rs_list_angular, (distance_and_angle.shape[0], len(self.theta_list)))\
            .reshape(-1, len(self.theta_list), len(self.Rs_list_angular), 1)

        
        # x3 speed up by using vectorized angular_symmetry_function
        triplet_x_thetas_x_Rs = LigandProteinComplex.angular_symmetry_function_vectorized(r_ij_mesh, r_ik_mesh, theta_jik_mesh, thetas_mesh, Rs_mesh, 6, 8, 4)

        element_x_thetas_x_Rs = np.apply_along_axis(
            lambda x: np.bincount(target_triplet_element_id, weights=x, minlength=len(target_elements_triplet)), 
            axis=0, arr=triplet_x_thetas_x_Rs
        )
        feature_value = element_x_thetas_x_Rs.reshape(-1)

        return feature_name, feature_value


class ExpOnionNet(LigandProteinComplex):
    def __init__(self, pdb_file, lig_code="LGD", target_elements=["H", "C", "N", "O", "P", "S", "Cl", "DU"]):
        super().__init__(pdb_file, lig_code=lig_code, target_elements=target_elements,)
        return None

    def generate_feature():

        return None