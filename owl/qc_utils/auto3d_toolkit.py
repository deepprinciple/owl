# ========= Copyright 2023-2024 @ DeepPrinciple. All Rights Reserved. =========
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ========= Copyright 2023-2024 @ DeepPrinciple. All Rights Reserved. =========

from typing import List, Dict, Any, Optional, Union
import os
import numpy as np
import tempfile

from camel.toolkits.base import BaseToolkit
from camel.toolkits.function_tool import FunctionTool

from Auto3D.auto3D import options, smiles2mols

# from yarp.wrappers.crest import CREST
# from yarp.wrappers.pyscf import PYSCF
# from yarp.wrappers.xtb import XTB
# from yarp.wrappers.pygsm import PyGSM
# from yarp.wrappers.pysis import PYSIS
# from yarp.parsers import xyz_parse, xyz_write


class Auto3DToolkit(BaseToolkit):
    r"""A toolkit for Auto3D molecular structure generation.

    This toolkit provides methods to generate 3D structures from SMILES strings
    using the Auto3D package (https://github.com/isayevlab/Auto3D_pkg).
    """

    def generate_3d_structure(
        self, 
        smiles: str, 
        work_folder: str = os.getcwd(),
        k: int = 1,
        verbose: bool = False,
        jobname: str = 'auto3d',
        enumerate_tautomer: bool = False,
        isomer_engine: str = 'rdkit',
        enumerate_isomer: bool = True,
        mpi_np: int = 4,
        use_gpu: bool = True,
        optimizing_engine: str = 'AIMNET',
        opt_steps: int = 5000,
        convergence_threshold: float = 0.005,
        save_xyz: bool = False
    ) -> Dict[str, Any]:
        r"""Generate 3D structure from a SMILES string.

        Args:
            smiles (str): SMILES string.
            work_folder (str, optional): Directory for running calculations. Defaults to current directory.
            k (int, optional): Outputs the top-k structures for each SMILES. Defaults to 1.
            verbose (bool, optional): When True, save all meta data while running. Defaults to False.
            jobname (str, optional): Base name for output files. Defaults to 'auto3d'.
            enumerate_tautomer (bool, optional): When True, enumerate tautomers for the input. Defaults to False.
            isomer_engine (str, optional): Engine for isomer generation ('rdkit' or 'omega'). Defaults to 'rdkit'.
            enumerate_isomer (bool, optional): When True, cis/trans and r/s isomers are enumerated. Defaults to True.
            mpi_np (int, optional): Number of CPU cores for isomer generation. Defaults to 4.
            use_gpu (bool, optional): Whether to use GPU for calculations. Defaults to True.
            optimizing_engine (str, optional): Engine for optimization ('ANI2x', 'ANI2xt', 'AIMNET'). Defaults to 'AIMNET'.
            opt_steps (int, optional): Maximum optimization steps. Defaults to 5000.
            convergence_threshold (float, optional): Convergence threshold. Defaults to 0.005.
            save_xyz (bool, optional): Whether to save the structure as an XYZ file. Defaults to False.

        Returns:
            Dict[str, Any]: Dictionary containing results of the calculation.
        """
        # Create output directory if it doesn't exist
        os.makedirs(work_folder, exist_ok=True)
        
        # Set up Auto3D options according to the actual function signature
        args = options(
            k=k, 
            window=False,
            verbose=verbose,
            job_name=jobname,
            enumerate_tautomer=enumerate_tautomer,
            isomer_engine=isomer_engine.lower(),
            enumerate_isomer=enumerate_isomer,
            mpi_np=mpi_np,
            use_gpu=use_gpu,
            optimizing_engine=optimizing_engine,
            opt_steps=opt_steps,
            convergence_threshold=convergence_threshold
        )
        
        try:
            mols = smiles2mols([smiles], args)
            success = True
            
            if len(mols) > 0:
                mol = mols[0]
                # conformer = mol.GetConformer()
                positions = mol.GetConformer().GetPositions()
                elements = [atom.GetSymbol() for atom in mol.GetAtoms()]
                
                # Get energy if available
                energy = mol.GetProp('Energy') if mol.HasProp('Energy') else None
                
                # Save as XYZ file if requested
                xyz_file = None
                if save_xyz:
                    xyz_file = os.path.join(work_folder, f"{jobname}.xyz")
                    self._xyz_write(xyz_file, elements, positions)
                
                structure = {
                    'smiles': smiles,
                    'coordinates': positions.tolist(),
                    'elements': elements,
                    'energy': energy,
                    'xyz_file': xyz_file
                }
            else:
                structure = None
                
        except Exception as e:
            success = False
            structure = None
            print(f"Auto3D generation failed: {str(e)}")
        
        results = {
            "success": success,
            "structure": structure
        }
        
        return results

    def _xyz_write(self, name, elements, geo, append_opt=False, comment=''):
        """
        Simple wrapper function for writing xyz file

        Args:
            name (str): Filename of the output
            elements (List[str]): List of element types
            geo (np.ndarray): Nx3 array holding the cartesian coordinates
            append_opt (bool, optional): Whether to append to existing file. Defaults to False.
            comment (str, optional): Comment line for the XYZ file. Defaults to ''.
        """
        open_cond = 'a' if append_opt else 'w'
            
        with open(name, open_cond) as f:
            f.write('{}\n'.format(len(elements)))
            f.write('{}\n'.format(comment))
            for count_i, i in enumerate(elements):
                f.write("{:<20s} {:< 20.8f} {:< 20.8f} {:< 20.8f}\n".format(
                    i, geo[count_i][0], geo[count_i][1], geo[count_i][2]))

    def get_tools(self) -> List[FunctionTool]:
        r"""Returns a list of FunctionTool objects for the Auto3D toolkit.

        Returns:
            List[FunctionTool]: A list of FunctionTool objects.
        """
        return [
            FunctionTool(self.generate_3d_structure),
        ]


# class CRESTToolkit(BaseToolkit):
#     r"""A toolkit for CREST conformer sampling.

#     This toolkit provides methods to perform conformer sampling using the CREST
#     wrapper in YARP.
#     """

#     def sample_conformers(
#         self,
#         input_geo: str,
#         work_folder: str = os.getcwd(),
#         lot: str = 'gfn2',
#         nproc: int = 1,
#         mem: int = 2000,
#         solvent: bool = False,
#         opt_level: str = 'tight',
#         solvation_model: str = 'alpb',
#         runtype: str = 'imtd-gc',
#         charge: int = 0,
#         multiplicity: int = 1,
#         quick_mode: bool = False
#     ) -> Dict[str, Any]:
#         r"""Sample conformers using CREST.

#         Args:
#             input_geo (str): Path to XYZ file containing input geometry.
#             work_folder (str, optional): Directory for running calculations. Defaults to current directory.
#             lot (str, optional): Level of theory. Defaults to 'gfn2'.
#             nproc (int, optional): Number of processors to use. Defaults to 1.
#             mem (int, optional): Memory in MB. Defaults to 2000.
#             solvent (bool, optional): Whether to use implicit solvent. Defaults to False.
#             opt_level (str, optional): Optimization level. Defaults to 'tight'.
#             solvation_model (str, optional): Solvation model. Defaults to 'alpb'.
#             runtype (str, optional): CREST run type. Defaults to 'imtd-gc'.
#             charge (int, optional): Molecular charge. Defaults to 0.
#             multiplicity (int, optional): Spin multiplicity. Defaults to 1.
#             quick_mode (bool, optional): Whether to use quick mode. Defaults to False.

#         Returns:
#             Dict[str, Any]: Dictionary containing results of the calculation.
#         """
#         crest = CREST(
#             input_geo=input_geo,
#             work_folder=work_folder,
#             lot=lot,
#             nproc=nproc,
#             mem=mem,
#             solvent=solvent,
#             opt_level=opt_level,
#             solvation_model=solvation_model,
#             runtype=runtype,
#             charge=charge,
#             multiplicity=multiplicity,
#             quick_mode=quick_mode
#         )
        
#         crest.execute()
        
#         results = {
#             "success": crest.calculation_terminated_normally(),
#             "output_file": crest.output,
#             "conformers": crest.get_conformers() if crest.calculation_terminated_normally() else None,
#             "energies": crest.get_energies() if crest.calculation_terminated_normally() else None
#         }
        
#         return results

#     def get_tools(self) -> List[FunctionTool]:
#         r"""Returns a list of FunctionTool objects for the CREST toolkit.

#         Returns:
#             List[FunctionTool]: A list of FunctionTool objects.
#         """
#         return [
#             FunctionTool(self.sample_conformers),
#         ]


# class PySCFToolkit(BaseToolkit):
#     r"""A toolkit for PySCF quantum chemistry calculations.

#     This toolkit provides methods to perform quantum chemistry calculations
#     using the PySCF wrapper in YARP.
#     """

#     def run_calculation(
#         self,
#         input_geo: str,
#         work_folder: str = os.getcwd(),
#         lot: str = 'b3lyp/def2svp',
#         jobtype: str = 'opt',
#         nproc: int = 1,
#         mem: int = 1000,
#         jobname: str = 'pyscfjob',
#         charge: int = 0,
#         multiplicity: int = 1,
#         solvation_model: str = None,
#         solvent_epi: float = 78.3553,
#         dispersion: str = None,
#         conv_tol: float = 1e-8,
#         max_cycle: int = 50
#     ) -> Dict[str, Any]:
#         r"""Run a PySCF calculation.

#         Args:
#             input_geo (str): Path to XYZ file containing input geometry.
#             work_folder (str, optional): Directory for running calculations. Defaults to current directory.
#             lot (str, optional): Level of theory. Defaults to 'b3lyp/def2svp'.
#             jobtype (str, optional): Type of calculation. Defaults to 'opt'.
#             nproc (int, optional): Number of processors to use. Defaults to 1.
#             mem (int, optional): Memory in MB. Defaults to 1000.
#             jobname (str, optional): Base name for output files. Defaults to 'pyscfjob'.
#             charge (int, optional): Molecular charge. Defaults to 0.
#             multiplicity (int, optional): Spin multiplicity. Defaults to 1.
#             solvation_model (str, optional): Solvation model. Defaults to None.
#             solvent_epi (float, optional): Solvent dielectric constant. Defaults to 78.3553.
#             dispersion (str, optional): Dispersion correction. Defaults to None.
#             conv_tol (float, optional): Convergence tolerance. Defaults to 1e-8.
#             max_cycle (int, optional): Maximum SCF cycles. Defaults to 50.

#         Returns:
#             Dict[str, Any]: Dictionary containing results of the calculation.
#         """
#         pyscf = PYSCF(
#             input_geo=input_geo,
#             work_folder=work_folder,
#             lot=lot,
#             jobtype=jobtype,
#             nproc=nproc,
#             mem=mem,
#             jobname=jobname,
#             charge=charge,
#             multiplicity=multiplicity,
#             solvation_model=solvation_model,
#             solvent_epi=solvent_epi,
#             dispersion=dispersion,
#             conv_tol=conv_tol,
#             max_cycle=max_cycle
#         )
        
#         pyscf.execute()
        
#         results = {
#             "success": pyscf.calculation_terminated_normally(),
#             "output_file": pyscf.output
#         }
        
#         if pyscf.calculation_terminated_normally():
#             if jobtype == 'opt' or jobtype == 'tsopt':
#                 results["optimization_converged"] = pyscf.optimization_converged()
#                 if results["optimization_converged"]:
#                     results["final_structure"] = pyscf.get_final_structure()
            
#             results["energy"] = pyscf.get_energy()
            
#             try:
#                 results["homo_lumo"] = pyscf.get_homo_lumo()
#             except:
#                 pass
                
#             try:
#                 results["gradients"] = pyscf.get_gradients()
#             except:
#                 pass
                
#             try:
#                 results["hessian"] = pyscf.get_hessian()
#             except:
#                 pass
        
#         return results

#     def get_tools(self) -> List[FunctionTool]:
#         r"""Returns a list of FunctionTool objects for the PySCF toolkit.

#         Returns:
#             List[FunctionTool]: A list of FunctionTool objects.
#         """
#         return [
#             FunctionTool(self.run_calculation),
#         ]


# class XTBToolkit(BaseToolkit):
#     r"""A toolkit for XTB quantum chemistry calculations.

#     This toolkit provides methods to perform semi-empirical quantum chemistry
#     calculations using the XTB wrapper in YARP.
#     """

#     def run_calculation(
#         self,
#         input_geo: str,
#         work_folder: str = os.getcwd(),
#         lot: str = 'gfn2',
#         jobtype: List[str] = ['sp'],
#         nproc: int = 1,
#         scf_iters: int = 250,
#         jobname: str = 'xtbjob',
#         solvent: bool = False,
#         solvation_model: str = 'alpb',
#         charge: int = 0,
#         multiplicity: int = 1
#     ) -> Dict[str, Any]:
#         r"""Run an XTB calculation.

#         Args:
#             input_geo (str): Path to XYZ file containing input geometry.
#             work_folder (str, optional): Directory for running calculations. Defaults to current directory.
#             lot (str, optional): Level of theory. Defaults to 'gfn2'.
#             jobtype (List[str], optional): Types of calculations to perform. Defaults to ['sp'].
#             nproc (int, optional): Number of processors to use. Defaults to 1.
#             scf_iters (int, optional): Maximum SCF iterations. Defaults to 250.
#             jobname (str, optional): Base name for output files. Defaults to 'xtbjob'.
#             solvent (bool, optional): Whether to use implicit solvation. Defaults to False.
#             solvation_model (str, optional): Type of solvation model. Defaults to 'alpb'.
#             charge (int, optional): Molecular charge. Defaults to 0.
#             multiplicity (int, optional): Spin multiplicity. Defaults to 1.

#         Returns:
#             Dict[str, Any]: Dictionary containing results of the calculation.
#         """
#         xtb = XTB(
#             input_geo=input_geo,
#             work_folder=work_folder,
#             lot=lot,
#             jobtype=jobtype,
#             nproc=nproc,
#             scf_iters=scf_iters,
#             jobname=jobname,
#             solvent=solvent,
#             solvation_model=solvation_model,
#             charge=charge,
#             multiplicity=multiplicity
#         )
        
#         xtb.execute()
        
#         results = {
#             "success": xtb.calculation_terminated_normally(),
#             "output_file": xtb.output
#         }
        
#         if xtb.calculation_terminated_normally():
#             results["energy"] = xtb.get_energy()
            
#             if 'opt' in jobtype:
#                 results["optimization_success"] = xtb.optimization_success()
#                 if results["optimization_success"]:
#                     results["final_structure"] = xtb.get_final_structure()
            
#             if 'hess' in jobtype:
#                 try:
#                     results["hessian"] = xtb.get_hessian()
#                     results["vibrational_frequencies"] = xtb.get_frequencies()
#                 except:
#                     pass
            
#             if 'grad' in jobtype:
#                 try:
#                     results["gradients"] = xtb.get_gradients()
#                 except:
#                     pass
        
#         return results

#     def get_tools(self) -> List[FunctionTool]:
#         r"""Returns a list of FunctionTool objects for the XTB toolkit.

#         Returns:
#             List[FunctionTool]: A list of FunctionTool objects.
#         """
#         return [
#             FunctionTool(self.run_calculation),
#         ]


# class PyGSMToolkit(BaseToolkit):
#     r"""A toolkit for PyGSM reaction path calculations.

#     This toolkit provides methods to perform reaction path calculations
#     using the PyGSM wrapper in YARP.
#     """

#     def find_reaction_path(
#         self,
#         reactant_xyz: str,
#         product_xyz: str,
#         work_folder: str = os.getcwd(),
#         calc: str = 'xtb',
#         nproc: int = 1,
#         ID: int = 0,
#         charge: int = 0,
#         multiplicity: int = 1,
#         max_gsm_iters: int = 100,
#         max_opt_iters: int = 100,
#         restart_file: str = None,
#         gsm_type: str = 'DE_GSM'
#     ) -> Dict[str, Any]:
#         r"""Find a reaction path using PyGSM.

#         Args:
#             reactant_xyz (str): Path to XYZ file containing reactant geometry.
#             product_xyz (str): Path to XYZ file containing product geometry.
#             work_folder (str, optional): Directory for running calculations. Defaults to current directory.
#             calc (str, optional): Calculator to use. Defaults to 'xtb'.
#             nproc (int, optional): Number of processors to use. Defaults to 1.
#             ID (int, optional): Job ID. Defaults to 0.
#             charge (int, optional): Molecular charge. Defaults to 0.
#             multiplicity (int, optional): Spin multiplicity. Defaults to 1.
#             max_gsm_iters (int, optional): Maximum GSM iterations. Defaults to 100.
#             max_opt_iters (int, optional): Maximum optimization iterations. Defaults to 100.
#             restart_file (str, optional): Path to restart file. Defaults to None.
#             gsm_type (str, optional): Type of GSM calculation. Defaults to 'DE_GSM'.

#         Returns:
#             Dict[str, Any]: Dictionary containing results of the calculation.
#         """
#         pygsm = PyGSM(
#             reactant_xyz=reactant_xyz,
#             product_xyz=product_xyz,
#             work_folder=work_folder,
#             calc=calc,
#             nproc=nproc,
#             ID=ID,
#             charge=charge,
#             multiplicity=multiplicity,
#             max_gsm_iters=max_gsm_iters,
#             max_opt_iters=max_opt_iters,
#             restart_file=restart_file,
#             gsm_type=gsm_type
#         )
        
#         pygsm.execute()
        
#         results = {
#             "success": pygsm.calculation_terminated_normally(),
#             "output_file": pygsm.output
#         }
        
#         if pygsm.calculation_terminated_normally():
#             results["reaction_path"] = pygsm.get_reaction_path()
#             results["energies"] = pygsm.get_energies()
#             results["ts_structure"] = pygsm.get_ts_structure()
        
#         return results

#     def get_tools(self) -> List[FunctionTool]:
#         r"""Returns a list of FunctionTool objects for the PyGSM toolkit.

#         Returns:
#             List[FunctionTool]: A list of FunctionTool objects.
#         """
#         return [
#             FunctionTool(self.find_reaction_path),
#         ]


# class PySISToolkit(BaseToolkit):
#     r"""A toolkit for PySIS transition state optimization.

#     This toolkit provides methods to perform transition state optimization
#     using the PySIS wrapper in YARP.
#     """

#     def optimize_transition_state(
#         self,
#         input_geo: str,
#         work_folder: str = os.getcwd(),
#         jobname: str = 'pysis',
#         jobtype: str = 'tsopt',
#         coord_type: str = 'redund',
#         calctype: str = 'xtb',
#         nproc: int = 1,
#         mem: int = 1000,
#         charge: int = 0,
#         multiplicity: int = 1,
#         functional: str = 'b3lyp',
#         basis: str = 'def2svp',
#         solvation_model: str = None,
#         solvent_epi: float = 78.3553,
#         dispersion: str = None
#     ) -> Dict[str, Any]:
#         r"""Optimize a transition state using PySIS.

#         Args:
#             input_geo (str): Path to XYZ file containing input geometry.
#             work_folder (str, optional): Directory for running calculations. Defaults to current directory.
#             jobname (str, optional): Base name for output files. Defaults to 'pysis'.
#             jobtype (str, optional): Type of calculation. Defaults to 'tsopt'.
#             coord_type (str, optional): Type of coordinates. Defaults to 'redund'.
#             calctype (str, optional): Type of calculator. Defaults to 'xtb'.
#             nproc (int, optional): Number of processors to use. Defaults to 1.
#             mem (int, optional): Memory in MB. Defaults to 1000.
#             charge (int, optional): Molecular charge. Defaults to 0.
#             multiplicity (int, optional): Spin multiplicity. Defaults to 1.
#             functional (str, optional): DFT functional. Defaults to 'b3lyp'.
#             basis (str, optional): Basis set. Defaults to 'def2svp'.
#             solvation_model (str, optional): Solvation model. Defaults to None.
#             solvent_epi (float, optional): Solvent dielectric constant. Defaults to 78.3553.
#             dispersion (str, optional): Dispersion correction. Defaults to None.

#         Returns:
#             Dict[str, Any]: Dictionary containing results of the calculation.
#         """
#         pysis = PYSIS(
#             input_geo=input_geo,
#             work_folder=work_folder,
#             jobname=jobname,
#             jobtype=jobtype,
#             coord_type=coord_type,
#             nproc=nproc,
#             mem=mem,
#             charge=charge,
#             multiplicity=multiplicity,
#             functional=functional,
#             basis=basis,
#             solvation_model=solvation_model,
#             solvent_epi=solvent_epi,
#             dispersion=dispersion
#         )
        
#         pysis.generate_input(calctype=calctype)
#         pysis.execute()
        
#         results = {
#             "success": pysis.calculation_terminated_normally(),
#             "output_file": pysis.output
#         }
        
#         if pysis.calculation_terminated_normally():
#             results["optimization_converged"] = pysis.optimization_converged()
#             if results["optimization_converged"]:
#                 results["final_structure"] = pysis.get_final_structure()
#                 results["energy"] = pysis.get_energy()
                
#                 try:
#                     results["hessian"] = pysis.get_hessian()
#                     results["vibrational_frequencies"] = pysis.get_frequencies()
#                 except:
#                     pass
        
#         return results

#     def get_tools(self) -> List[FunctionTool]:
#         r"""Returns a list of FunctionTool objects for the PySIS toolkit.

#         Returns:
#             List[FunctionTool]: A list of FunctionTool objects.
#         """
#         return [
#             FunctionTool(self.optimize_transition_state),
#         ]


# class QuantumChemToolkit(BaseToolkit):
#     r"""A comprehensive toolkit for quantum chemistry calculations.

#     This toolkit combines all the individual quantum chemistry toolkits
#     to provide a unified interface for quantum chemistry calculations.
#     """

#     def __init__(self):
#         r"""Initialize the QuantumChemToolkit."""
#         self.auto3d_toolkit = Auto3DToolkit()
#         self.crest_toolkit = CRESTToolkit()
#         self.pyscf_toolkit = PySCFToolkit()
#         self.xtb_toolkit = XTBToolkit()
#         self.pygsm_toolkit = PyGSMToolkit()
#         self.pysis_toolkit = PySISToolkit()

#     def get_tools(self) -> List[FunctionTool]:
#         r"""Returns a list of FunctionTool objects from all the individual toolkits.

#         Returns:
#             List[FunctionTool]: A list of FunctionTool objects.
#         """
#         tools = []
#         tools.extend(self.auto3d_toolkit.get_tools())
#         tools.extend(self.crest_toolkit.get_tools())
#         tools.extend(self.pyscf_toolkit.get_tools())
#         tools.extend(self.xtb_toolkit.get_tools())
#         tools.extend(self.pygsm_toolkit.get_tools())
#         tools.extend(self.pysis_toolkit.get_tools())
#         return tools 