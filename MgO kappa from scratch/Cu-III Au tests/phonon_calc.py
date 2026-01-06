import ase
from ase.build import bulk
from ase.calculators.lammpslib import LAMMPSlib
from kaldo.forceconstants import ForceConstants
import numpy as np

import logging

logging.basicConfig(level=logging.INFO)

FNAME = "cu_relaxed"
atoms = ase.io.lammpsdata.read_lammps_data(FNAME, atom_style="atomic")