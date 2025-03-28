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

from typing import List, Dict, Any, Optional, Union, Tuple
import os
import numpy as np

from camel.toolkits.base import BaseToolkit
from camel.toolkits.function_tool import FunctionTool

from yarp.wrappers.xtb import XTB


class XTBToolkit(BaseToolkit):
    r"""A toolkit for xTB quantum chemistry calculations.

    This toolkit provides an interface to run various types of calculations using the 
    extended tight-binding (xTB) quantum chemistry method, including single point energy,
    geometry optimization, gradient and Hessian calculations, with optional implicit solvation.
    
    The xTB method is a semi-empirical quantum chemistry method that provides a good 
    balance between accuracy and computational efficiency, making it suitable for 
    large molecules and high-throughput screening.
    
    Example usage:
    ```python
    # Run a geometry optimization with GFN2-xTB
    result = run_calculation(
        input_geo="molecule.xyz",
        work_folder="./calculations",
        lot="gfn2",
        jobtype=["opt"],
        nproc=4
    )
    
    # Check if optimization was successful and get results
    if result["success"] and result["optimization_success"]:
        print(f"Final energy: {result['energy']} Hartree")
        print(f"Optimized structure: {result['final_structure']}")
    ```
    """

    def run_calculation(
        self,
        input_geo: str,
        work_folder: str = os.getcwd(),
        lot: str = 'gfn2',
        jobtype: List[str] = ['opt'],
        nproc: int = 1,
        scf_iters: int = 300,
        jobname: str = 'xtbjob',
        solvent: Union[bool, str] = False,
        solvation_model: str = 'alpb',
        charge: int = 0,
        multiplicity: int = 1,
        distance_constraints: List[List] = None,
        cartesian_constraints: List[int] = None,
        force_constant: float = 0.5,
        additional_commands: str = None,
        clean_up: bool = False,
    ) -> Dict[str, Any]:
        r"""Run an xTB calculation.

        This function provides a comprehensive interface to the xTB quantum chemistry program,
        allowing for various types of calculations including single point energy, geometry
        optimization, and frequency analysis.

        Args:
            input_geo (str): Path to XYZ file containing input geometry. Must be a valid XYZ file.
            work_folder (str, optional): Directory for running calculations and saving output files.
                Will be created if it doesn't exist. Defaults to current directory.
            lot (str, optional): Level of theory to use. Options include:
                - 'gfn1': GFN1-xTB method
                - 'gfn2': GFN2-xTB method (recommended for most cases)
                - 'gfnff': Force-field based method for very large systems
                Defaults to 'gfn2'.
            jobtype (List[str], optional): Types of calculations to perform. Options include:
                - 'sp': Single point energy calculation
                - 'opt': Geometry optimization
                - 'grad': Gradient calculation
                - 'hess': Hessian calculation (for vibrational frequencies)
                Multiple job types can be combined, e.g., ['opt', 'hess']. Defaults to ['opt'].
            nproc (int, optional): Number of processors to use for parallel calculations.
                Defaults to 1.
            scf_iters (int, optional): Maximum number of SCF iterations. Defaults to 300.
            jobname (str, optional): Base name for output files. Defaults to 'xtbjob'.
            solvent (Union[bool, str], optional): Solvent to use for implicit solvation.
                If False, no solvation is used. If True, water is used as default.
                Can also be a string specifying the solvent name. Defaults to False.
            solvation_model (str, optional): Type of solvation model to use. Options:
                - 'alpb': Analytical Linearized Poisson-Boltzmann (supports more solvents)
                - 'gbsa': Generalized Born Surface Area
                Defaults to 'alpb'.
            charge (int, optional): Molecular charge. Defaults to 0.
            multiplicity (int, optional): Spin multiplicity. Defaults to 1.
            distance_constraints (List[List], optional): List of distance constraints.
                Each element should be [atom_i, atom_j, distance] where atom indices start from 1.
                If distance is omitted, 'auto' will be used. Defaults to None.
            cartesian_constraints (List[int], optional): List of atom indices to constrain in Cartesian space.
                Indices start from 1. Defaults to None.
            force_constant (float, optional): Force constant for constraints in atomic units.
                Defaults to 0.5.
            additional_commands (str, optional): Additional command line arguments to pass to xTB.
                Defaults to None.
            clean_up (bool, optional): Whether to clean up temporary files after calculation.
                Defaults to False.

        Returns:
            Dict[str, Any]: Dictionary containing results of the calculation, including:
                - success (bool): Whether the calculation completed successfully
                - output_file (str): Path to the output file
                - energy (float, optional): Final energy in Hartree
                - homo_lumo_gap (float, optional): HOMO-LUMO gap in eV
                - optimization_converged (bool, optional): Whether optimization converged
                - optimization_success (bool, optional): Whether optimization produced a valid structure
                - final_structure (Dict, optional): Final optimized structure with elements and coordinates
                - final_xyz_file (str, optional): Path to the final XYZ file (if save_xyz=True)
                - charges (List[float], optional): Atomic charges
                - bond_orders (Dict[str, float], optional): Wiberg bond orders
                - gradients (List[List[float]], optional): Energy gradients
                - hessian (List[List[float]], optional): Hessian matrix
                - imaginary_frequency (float, optional): Imaginary frequency (if present)

        Example:
            ```python
            # Run a geometry optimization with GFN2-xTB in water
            result = run_calculation(
                input_geo="molecule.xyz",
                work_folder="./calculations",
                lot="gfn2",
                jobtype=["opt", "hess"],
                nproc=4,
                solvent="water",
                solvation_model="alpb",
                save_xyz=True
            )
            
            # Check results
            if result["success"]:
                print(f"Energy: {result['energy']} Hartree")
                if "imaginary_frequency" in result:
                    print(f"Imaginary frequency: {result['imaginary_frequency']} cm^-1")
            ```
        """
        # Initialize XTB object
        xtb = XTB(
            input_geo=input_geo,
            work_folder=work_folder,
            lot=lot,
            jobtype=jobtype,
            nproc=nproc,
            scf_iters=scf_iters,
            jobname=jobname,
            solvent=solvent,
            solvation_model=solvation_model,
            charge=charge,
            multiplicity=multiplicity
        )
        
        # Add additional commands and constraints
        xtb.add_command(
            additional=additional_commands,
            distance_constraints=distance_constraints,
            cartesian_constraints=cartesian_constraints,
            force_constant=force_constant
        )
        
        # Execute the calculation
        try:
            xtb.execute()
            success = xtb.calculation_terminated_normally()
        except Exception as e:
            success = False
            print(f"XTB calculation failed: {str(e)}")
        
        # Prepare results dictionary
        results = {
            "success": success,
            "output_file": xtb.output
        }
        
        # Add additional results if calculation was successful
        if success:
            # Add energy
            energy = xtb.get_energy()
            if energy:
                results["energy"] = energy
            
            # Add HOMO-LUMO gap
            gap = xtb.get_gap()
            if gap:
                results["homo_lumo_gap"] = gap
            
            # Add optimization results if applicable
            if 'opt' in jobtype:
                results["optimization_converged"] = xtb.optimization_converged()
                results["optimization_success"] = xtb.optimization_success()
                
                if results["optimization_success"]:
                    structure = xtb.get_final_structure()
                    if structure:
                        elements, coords = structure
                        results["final_structure"] = {
                            "elements": elements,
                            "coordinates": coords.tolist() if isinstance(coords, np.ndarray) else coords
                        }
            
            # Add charges if available
            charges = xtb.get_charge()
            if isinstance(charges, np.ndarray):
                results["charges"] = charges.tolist()
            
            # Add Wiberg bond orders if available
            wbo = xtb.get_wbo()
            if wbo:
                # Convert keys to strings for JSON serialization
                results["bond_orders"] = {f"{k[0]}-{k[1]}": v for k, v in wbo.items()}
            
            # Add gradient if applicable
            if 'grad' in jobtype:
                gradients = xtb.get_gradients()
                if isinstance(gradients, np.ndarray):
                    results["gradients"] = gradients.tolist()
            
            # Add Hessian and vibrational data if applicable
            if 'hess' in jobtype:
                hessian = xtb.get_hessian()
                if isinstance(hessian, np.ndarray):
                    results["hessian"] = hessian.tolist()
                
                imag_freq = xtb.get_imag_freq()
                if imag_freq:
                    results["imaginary_frequency"] = imag_freq
        
        # Clean up temporary files if requested
        if clean_up:
            xtb.clean_up(remove_output=False)
        
        return results

    def get_tools(self) -> List[FunctionTool]:
        r"""Returns a list of FunctionTool objects for the XTB toolkit.

        Returns:
            List[FunctionTool]: A list of FunctionTool objects.
        """
        return [
            FunctionTool(self.run_calculation),
        ]