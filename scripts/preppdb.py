import sys
import os
import numpy as np
import pandas as pd
from collections import defaultdict
from openbabel import pybel
from multiprocessing import Pool
from tqdm import tqdm
import tempfile

class rewritePDB(object):
    """
    Modify pdb file by changing atom indexing, resname, res sequence number and chain id

    Parameters
    ----------

    Attributes
    ----------

    """
    def __init__(self, inpdb):
        self.pdb = inpdb

    def pdbRewrite(self, input, output, chain, atomStartNdx, resStartNdx):
        """
        change atom id, residue id and chain id
        :param input: str, input pdb file
        :param output: str, output file name
        :param chain: str, chain id
        :param atomStartNdx: int,
        :param resStartNdx: int
        :return:
        """
        resseq = int(resStartNdx)
        atomseq = int(atomStartNdx)
        chainname = chain

        newfile = open(output,'w')
        resseq_list = []

        try :
            with open(input) as lines :
                for s in lines :
                    if "ATOM" in s and len(s.split()) > 6 :
                        atomseq += 1
                        newline = s
                        newline = self.atomSeqChanger(newline, atomseq)
                        newline = self.chainIDChanger(newline, chainname)
                        if len(resseq_list) == 0 :
                            newline = self.resSeqChanger(newline, resseq)
                            resseq_list.append(int(s[22:26].strip()))
                        else :
                            if resseq_list[-1] == int(s[22:26].strip()) :
                                newline = self.resSeqChanger(newline, resseq)
                            else :
                                resseq += 1
                                newline = self.resSeqChanger(newline, resseq)
                            resseq_list.append(int(s[22:26].strip()))
                        newfile.write(newline)
                    else :
                        newfile.write(s)
        except FileExistsError :
            print("File %s not exist" % self.pdb)

        newfile.close()
        return 1

    def resSeqChanger(self, inline, resseq):
        resseqstring = " "*(4 - len(str(resseq)))+str(resseq)
        newline = inline[:22] + resseqstring + inline[26:]
        return newline

    def atomSeqChanger(self, inline, atomseq):
        atomseqstring = " " * (5 - len(str(atomseq))) + str(atomseq)
        newline = inline[:6] + atomseqstring + inline[11:]
        return newline

    def resNameChanger(self, inline, resname):
        resnamestr = " " * (4 - len(str(resname))) + str(resname)
        newline = inline[:16] + resnamestr + inline[20:]
        return newline

    def chainIDChanger(self, inline, chainid) :
        newline = inline[:21] + str(chainid) + inline[22:]
        return newline

    def atomNameChanger(self, inline, new_atom_name):
        newline = inline[:12] + "%4s" % new_atom_name + inline[16:]
        return newline

    def combinePDBFromLines(self, output, lines):
        """
        combine a list of lines to a pdb file

        Parameters
        ----------
        output
        lines

        Returns
        -------

        """

        with open(output, "wb") as tofile :
            tmp = map(lambda x: tofile.write(x), lines)

        return 1

    def swampPDB(self, input, atomseq_pdb, out_pdb, chain="B"):
        """
        given a pdb file (with coordinates in a protein pocket), but with wrong atom
        sequence order, try to re-order the pdb for amber topology building

        Parameters
        ----------
        input:str,
            the pdb file with the correct coordinates
        atomseq_pdb:str,
            the pdb file with correct atom sequences
        out_pdb: str,
            output pdb file name
        chain: str, default is B
            the chain identifier of a molecule

        Returns
        -------

        """

        tofile = open("temp.pdb", 'w')

        crd_list = {}

        ln_target, ln_source = 0, 0
        # generate a dict { atomname: pdbline}
        with open(input) as lines :
            for s in [x for x in lines if ("ATOM" in x or "HETATM" in x)]:
                crd_list[s.split()[2]] = s
                ln_source += 1

        # reorder the crd_pdb pdblines, according to the atomseq_pdb lines
        with open(atomseq_pdb) as lines:
            for s in [x for x in lines if ("ATOM" in x or "HETATM" in x)]:
                newline = crd_list[s.split()[2]]
                tofile.write(newline)
                ln_target += 1

        tofile.close()

        if ln_source != ln_target:
            print("Error: Number of lines in source and target pdb files are not equal. (%s %s)"%(input, atomseq_pdb))

        # re-sequence the atom index
        self.pdbRewrite(input="temp.pdb", atomStartNdx=1, chain=chain, output=out_pdb, resStartNdx=1)

        os.remove("temp.pdb")

        return None

def change_lig_name(ligand_input, ligand_output, lig_code='LGD'):

    pio = rewritePDB(ligand_input)
    tofile = open(ligand_output, "w")
    with open(ligand_input) as lines:
        for s in lines:
            if len(s.split()) and s.split()[0] in ['ATOM', 'HETATM']:
                nl = pio.resNameChanger(s, lig_code)
                #n2 = pio.chainIDChanger(nl, "Z")
                tofile.write(nl)

    tofile.close()
    return None

def convert_mol2_to_pdb(input_file, output_file):
    lig_mol = next(iter(pybel.readfile('mol2', input_file)))
    lig_mol.write('pdb', output_file, overwrite=True)
    return None

def combine_protein_and_ligand(protein_file, ligand_file, output_file):
    with open(protein_file) as p_f:
        protein = p_f.read()

    with open(ligand_file) as l_f:
        ligand = l_f.read()

    with open(output_file, mode='w') as o_f:
        o_f.write(protein + ligand)

    return None

def make_complex_pdb(pdb_input_dir, output, lig_code='LGD'):
    # retrieve pdb id from directory path
    pdb_id = os.path.basename(pdb_input_dir)

    # convert ligand mol2 file to pdb file
    ligand_mol2 = os.path.join(pdb_input_dir, f"{pdb_id}_ligand.mol2")
    ligand_pdb = os.path.join(pdb_input_dir, f"{pdb_id}_ligand.pdb")
    convert_mol2_to_pdb(ligand_mol2, ligand_pdb)

    # rename residue name of ligand pdb file
    ligand_pdb_renamed = os.path.join(pdb_input_dir, f"{pdb_id}_ligand_renamed.pdb")
    change_lig_name(ligand_pdb, ligand_pdb_renamed, lig_code=lig_code)

    # combine protein pdb and ligand_reneamed pdb
    protein_pdb = os.path.join(pdb_input_dir, f"{pdb_id}_protein.pdb")
    combine_protein_and_ligand(protein_pdb, ligand_pdb_renamed, output)
    
    return None

def _prepare_pdb(pdb_input_dir, output_dir, lig_code='LGD', output_suffix='complex'):
    pdb_id = os.path.basename(pdb_input_dir)
    output_path = os.path.join(output_dir, f"{pdb_id}_{output_suffix}.pdb")
    make_complex_pdb(pdb_input_dir, output_path, lig_code=lig_code)
    return None

def _wrapped_prepare_pdb(kwargs):
    _prepare_pdb(**kwargs)
    return None
    
def prepare_pdbs_from_pdbbind(pdb_dir_path_list, output_dir, lig_code='LGD', output_suffix='complex'):
    for pdb_input_dir in pdb_dir_path_list:
        _prepare_pdb(pdb_input_dir, output_dir, lig_code=lig_code, output_suffix=output_suffix)
    print('finished')
    return None

def prepare_pdbs_from_pdbbind_parallelized(pdb_dir_path_list, output_dir, num_cpu=1, lig_code='LGD', output_suffix='complex'):

    _prepare_pdb_args = [
        {'pdb_input_dir': pdb_input_dir, 
        'output_dir': output_dir, 
        'lig_code': lig_code, 
        'output_suffix': output_suffix}
        for pdb_input_dir in pdb_dir_path_list
    ]
    
    with Pool(processes=num_cpu) as p:
        list(tqdm(
            p.imap(_wrapped_prepare_pdb, _prepare_pdb_args),
            total=len(_prepare_pdb_args)
        ))
    print('finished')
    return None


######################## Docking decoy Preparation #########################################

def _prepare_docking_decoy_pdb(input_decoys_file, input_protein_pdb_file, 
                        output_complex_dir, lig_code='LGD'):
                        
    decoys = pybel.readfile('mol2', input_decoys_file)
    for decoy in decoys:
        # read decoy ligand mol2 and write to pdb file
        with tempfile.NamedTemporaryFile(suffix='.pdb') as temp_decoy_pdb:
            decoy.write('pdb', temp_decoy_pdb.name, overwrite=True)

            # rename residue name of decoy ligand pdb file
            with tempfile.NamedTemporaryFile(suffix='.pdb') as temp_decoy_pdb_renamed:
                change_lig_name(
                    temp_decoy_pdb.name, 
                    temp_decoy_pdb_renamed.name, 
                    lig_code=lig_code
                )
            
                # combine protein pdb and ligand_reneamed pdb
                output_complex_file = os.path.join(
                    output_complex_dir, 
                    f"{decoy.title}_complex.pdb"
                )
                combine_protein_and_ligand(
                    input_protein_pdb_file, 
                    temp_decoy_pdb_renamed.name, 
                    output_complex_file
                )
    return None
        
def _wrapped_prepare_docking_decoy_pdb(kwargs):
    output_complex_dir = kwargs['output_complex_dir']
    if not os.path.exists(output_complex_dir):
        os.makedirs(output_complex_dir)
    _prepare_docking_decoy_pdb(**kwargs)
    return None

def get_pdbid_from_path(input_decoys_file):
    return os.path.basename(input_decoys_file).replace('_decoys.mol2', '')

def prepare_docking_decoys_parallelized(
        input_decoys_files, input_protein_pdb_base_dir, 
        output_complex_base_dir, num_cpu=1, lig_code='LGD'):
    
    _prepare_docking_decoy_pdb_args = [{
        'input_decoys_file': input_decoys_file, 
        'input_protein_pdb_file':  os.path.join(
            input_protein_pdb_base_dir, 
            get_pdbid_from_path(input_decoys_file), 
            f"{get_pdbid_from_path(input_decoys_file)}_protein.pdb"
        ), 
        'output_complex_dir': os.path.join(
            output_complex_base_dir, 
            get_pdbid_from_path(input_decoys_file)
        ),
        'lig_code': lig_code
        }
        for input_decoys_file in input_decoys_files
    ]

    with Pool(processes=num_cpu) as p:
        list(tqdm(
            p.imap(_wrapped_prepare_docking_decoy_pdb, _prepare_docking_decoy_pdb_args),
            total=len(_prepare_docking_decoy_pdb_args)
        ))
    return None


######################## Screening decoy Preparation #########################################


def _prepare_screening_decoy_pdb(input_decoys_file, input_protein_pdb_file, 
                        output_dir, lig_code='LGD'):
    
    prt_pdb_id, lig_pdb_id = os.path.splitext(os.path.basename(input_decoys_file))[0].split('_')

    output_complex_dir = os.path.join(output_dir, prt_pdb_id, lig_pdb_id)
    if not os.path.exists(output_complex_dir):
        os.makedirs(output_complex_dir)

    decoys = pybel.readfile('mol2', input_decoys_file)
    for decoy in decoys:
        # read decoy ligand mol2 and write to pdb file
        with tempfile.NamedTemporaryFile(suffix='.pdb') as temp_decoy_pdb:
            decoy.write('pdb', temp_decoy_pdb.name, overwrite=True)

            # rename residue name of decoy ligand pdb file
            with tempfile.NamedTemporaryFile(suffix='.pdb') as temp_decoy_pdb_renamed:
                change_lig_name(
                    temp_decoy_pdb.name, 
                    temp_decoy_pdb_renamed.name, 
                    lig_code=lig_code
                )
            
                # combine protein pdb and ligand_reneamed pdb
                output_complex_file = os.path.join(
                    output_complex_dir, 
                    f"{prt_pdb_id}_{decoy.title}_complex.pdb"
                )
                combine_protein_and_ligand(
                    input_protein_pdb_file, 
                    temp_decoy_pdb_renamed.name, 
                    output_complex_file
                )
    return None

        
def _wrapped_prepare_screening_decoy_pdb(kwargs):
    _prepare_screening_decoy_pdb(**kwargs)
    return None

def get_protein_pdbid_from_path(input_decoys_file):
    return os.path.splitext(os.path.basename(input_decoys_file))[0].split('_')[0]

def prepare_screening_decoys_parallelized(
        input_decoys_files, input_protein_pdb_base_dir, 
        output_dir, num_cpu=1, lig_code='LGD'):

    """
    example:
        input_decoys_files = glob.glob('/home/AQDNet/1.Input/1.RawDataset/CASF-2016/decoys_screening/1e66/*.mol2')
        input_protein_pdb_base_dir = '/home/AQDNet/1.Input/1.RawDataset/CASF-2016/coreset/'
        output_dir = '/home/AQDNet/1.Input/2.PreprocessedPDB/CASF2016/decoys_screening/'
        num_cpu = 5
        prepare_screening_decoys_parallelized(
                input_decoys_files, input_protein_pdb_base_dir, 
                output_dir, num_cpu=num_cpu
        )

    """
    
    _prepare_screening_decoy_pdb_args = [{
        'input_decoys_file': input_decoys_file, 
        'input_protein_pdb_file':  os.path.join(
            input_protein_pdb_base_dir, 
            get_protein_pdbid_from_path(input_decoys_file), 
            f"{get_protein_pdbid_from_path(input_decoys_file)}_protein.pdb"
        ), 
        'output_dir': output_dir,
        'lig_code': lig_code
        }
        for input_decoys_file in input_decoys_files
    ]

    with Pool(processes=num_cpu) as p:
        list(tqdm(
            p.imap(_wrapped_prepare_screening_decoy_pdb, _prepare_screening_decoy_pdb_args),
            total=len(_prepare_screening_decoy_pdb_args)
        ))
    return None

