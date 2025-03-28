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
import tempfile

from camel.toolkits.base import BaseToolkit
from camel.toolkits.function_tool import FunctionTool

from yarp.parsers import xyz_parse, xyz_write, mol_write, to_xyz_string


class MoleculeIOToolkit(BaseToolkit):
    r"""A toolkit for molecular file I/O operations.

    This toolkit provides utilities for reading, writing, and manipulating
    molecular structure files in various formats (XYZ, MOL, etc.).
    """

    def create_xyz_file(
        self,
        elements: List[str],
        coordinates: List[List[float]],
        output_file: str = None,
        comment: str = "",
        append_opt: bool = False
    ) -> str:
        r"""Create an XYZ file from atomic elements and coordinates.

        This function creates a standard XYZ file that can be used as input for quantum chemistry calculations.

        Args:
            elements (List[str]): List of atomic elements (e.g., ['C', 'H', 'H', 'H', 'H'])
            coordinates (List[List[float]]): List of atomic coordinates in Angstroms
                Each coordinate should be a list of 3 floats [x, y, z]
            output_file (str, optional): Path where the XYZ file should be saved.
                If None, a temporary file will be created. Defaults to None.
            comment (str, optional): Comment line for the XYZ file. Defaults to "".
            append_opt (bool, optional): Whether to append to existing file. Defaults to False.

        Returns:
            str: Path to the created XYZ file
        """
        if output_file is None:
            temp_dir = tempfile.gettempdir()
            output_file = os.path.join(temp_dir, f"temp_{os.getpid()}.xyz")
        
        xyz_write(output_file, elements, np.array(coordinates), append_opt=append_opt, comment=comment)
        return output_file

    def parse_xyz_file(
        self, 
        xyz_file: str, 
        multiple: bool = False, 
        return_info: bool = False
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        r"""Parse an XYZ file to extract elements and coordinates.

        Args:
            xyz_file (str): Path to the XYZ file to parse
            multiple (bool, optional): Whether to parse multiple molecules from the file. Defaults to False.
            return_info (bool, optional): Whether to return the comment line. Defaults to False.

        Returns:
            Union[Dict[str, Any], List[Dict[str, Any]]]: 
                If multiple=False, returns a dictionary containing:
                    - "elements" (List[str]): List of atomic elements
                    - "coordinates" (List[List[float]]): List of atomic coordinates
                    - "comment" (str): Comment line from XYZ file (only if return_info=True)
                If multiple=True, returns a list of such dictionaries.
        """
        if return_info:
            result = xyz_parse(xyz_file, multiple=multiple, return_info=True)
            
            if multiple:
                molecules, comments = result
                return [
                    {
                        "elements": mol[0],
                        "coordinates": mol[1].tolist(),
                        "comment": comment
                    } for mol, comment in zip(molecules, comments)
                ]
            else:
                (elements, geo), comment = result
                return {
                    "elements": elements,
                    "coordinates": geo.tolist(),
                    "comment": comment
                }
        else:
            result = xyz_parse(xyz_file, multiple=multiple)
            
            if multiple:
                return [
                    {
                        "elements": mol[0],
                        "coordinates": mol[1].tolist()
                    } for mol in result
                ]
            else:
                elements, geo = result
                return {
                    "elements": elements,
                    "coordinates": geo.tolist()
                }

    def create_mol_file(
        self,
        elements: List[str],
        coordinates: List[List[float]],
        bond_matrix: List[List[int]],
        output_file: str,
        charge: int = 0,
        append_opt: bool = False
    ) -> str:
        r"""Create a MOL file (V2000 format) from molecular data.

        Args:
            elements (List[str]): List of atomic elements
            coordinates (List[List[float]]): List of atomic coordinates in Angstroms
            bond_matrix (List[List[int]]): NxN matrix representing bonds and bond orders
            output_file (str): Path where the MOL file should be saved
            charge (int, optional): Molecular charge. Defaults to 0.
            append_opt (bool, optional): Whether to append to existing file. Defaults to False.

        Returns:
            str: Path to the created MOL file
        """
        mol_write(output_file, elements, np.array(coordinates), np.array(bond_matrix), 
                 q=charge, append_opt=append_opt)
        return output_file

    def xyz_to_string(
        self, 
        elements: List[str], 
        coordinates: List[List[float]], 
        comment: str = ''
    ) -> str:
        r"""Convert molecular data to an XYZ format string.

        Args:
            elements (List[str]): List of atomic elements
            coordinates (List[List[float]]): List of atomic coordinates in Angstroms
            comment (str, optional): Comment line for the XYZ string. Defaults to ''.

        Returns:
            str: XYZ format string representation of the molecule
        """
        return to_xyz_string(elements, np.array(coordinates), comment=comment)

    def get_tools(self) -> List[FunctionTool]:
        r"""Returns a list of FunctionTool objects for the MoleculeIO toolkit.

        Returns:
            List[FunctionTool]: A list of FunctionTool objects.
        """
        return [
            FunctionTool(self.create_xyz_file),
            FunctionTool(self.parse_xyz_file),
            FunctionTool(self.create_mol_file),
            FunctionTool(self.xyz_to_string),
        ]
