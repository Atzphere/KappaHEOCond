#!/usr/bin/env python
# coding: utf-8

# ## Structure Generation
# The end result of these cells will be a set of computed force constants and everything needed to initialize a kALDO `Phonons` object. 

# For the purposes of MgO rock salt (a very harmonic crystal), we will define the structure with `ase` and perform MD calculations in LAMMPS to get the second- and third-order force constants. We will use start with a fit Buckingham potential from https://www.sciencedirect.com/science/article/pii/S0022311508003632?via%3Dihub

# ![image.png](attachment:15cb9482-2a2c-4189-907c-9b50e6258dc0.png)

# 3x4x5 dft supercell
# 4x4x4 x 3x4x5 md supercell

# In[1]:


from ase.build import bulk
from ase.calculators.lammpslib import LAMMPSlib
from kaldo.forceconstants import ForceConstants
import numpy as np

import logging

logging.basicConfig(level=logging.INFO)

# we will eventually need to ensure these lattice parameters are minimized
a = 4.212
atoms = bulk('MgO', 'rocksalt', a=a)

# as given by paper
atoms.set_initial_charges([1.7, -1.7])

# specify atom types and numbers for LAMMPS
atom_types = {'Mg': 1, 'O': 2}


# In[2]:


# from ase.visualize import view
# view(atoms)
print(atoms.get_initial_charges())
l1, l2, l3 = atoms.cell
norm = np.linalg.norm(l1)
print(f"cubic lattice constant: a = {norm * np.sqrt(2)}A")


# In[3]:


from kaldo.conductivity import Conductivity
from kaldo.phonons import Phonons
import pandas as pd

# replicate mgo paper

lammps_inputs = {
    'lammps_header': ['units metal',
    'atom_style charge', 'atom_modify map array sort 0 0'],

      'lmpcmds': [
          'kspace_style pppm 1.0e-6',
          'pair_style buck/coul/long 10.0',
          'pair_coeff 1 1 0 1 0',
          'pair_coeff 2 2 35686.18 0.201 32.0',
          'pair_coeff 1 2 929.69 0.29909 0.0'],

      'log_file': 'lammps-MgO-bulk.log',
      'keep_alive':True}

atoms.calc = LAMMPSlib(**lammps_inputs)

print("Energy ", atoms.get_potential_energy())


# In[4]:


from ase.filters import StrainFilter
from ase.optimize import BFGS
# relax structure
sf = StrainFilter(atoms)
dyn = BFGS(sf)


logging.info("Relaxing structure")
dyn.run(fmax=0.001)

l1, l2, l3 = atoms.cell
norm = np.linalg.norm(l1)
logging.info(f"cubic lattice constant: a = {norm * np.sqrt(2)}A")


# In[5]:


# buckingham potential 4x4x4 supercell
supercell = np.array([4, 4, 4])

# enlarged unit cell from paper
# supercell = np.array([4 * 3, 4 * 4, 4 * 5])

# enlarged unit cell from paper
supercell = np.array([3 * 3, 4 * 4, 3 * 5])

# Create a finite difference object
forceconstants_config  = {'atoms':atoms,'supercell': supercell,'folder':'force_constants'}
forceconstants = ForceConstants(**forceconstants_config)

# Compute 2nd and 3rd IFCs with LAMMPS using a tersoff potential
# delta_shift is how much to move atoms when computing forces
forceconstants.second.calculate(LAMMPSlib(**lammps_inputs), delta_shift=1e-3)
forceconstants.third.calculate(LAMMPSlib(**lammps_inputs), delta_shift=1e-3)


# In[6]:


# Create phonons object

# k-sampling grid for conductivity cals
k = 19
kpts = [k, k, k]
temperature = 300
is_classic = False # phonon mondes are treated as quantized
k_label = str(k) + '_' + str(k) + '_' + str(k)

phonons = Phonons(forceconstants=forceconstants,
                kpts=kpts,
                is_classic=is_classic,
                temperature=300,
                folder='si-bulk-ald-' + k_label,
                storage='numpy')


# ## Kappa calculation

# In[7]:


from kaldo.conductivity import Conductivity
qhgk_cond_matrix = Conductivity(phonons=phonons, method='qhgk').conductivity.sum(axis=0)

rta_cond_matrix = Conductivity(phonons=phonons, method='rta', n_interations=0).conductivity.sum(axis=0)

cond_rta = float(np.mean(np.diag(rta_cond_matrix)))

cond_qhgk = float(np.mean(np.diag(qhgk_cond_matrix)))

inv_cond_matrix = Conductivity(phonons=phonons, method='inverse').conductivity.sum(axis=0)
cond_inv = float(np.mean(np.diag(inv_cond_matrix)))


# In[8]:


print('RTA conductivity (W/mK): %.3f'%(cond_rta))
print(f'QHGK conductivity (W/mK): {cond_qhgk:.3f}')
print(f'inversion method conductivity (W/mK): {cond_inv:.3f}')


# * Strange discrepancy for what should be a very harmonic crystal
# * Possibly related to the structure not being perfectly relaxed?
# ![image.png](attachment:a09dea6e-4f60-46f3-a2b2-cc8962b4cf8b.png)

# ## Phonon Dispersion

# In[9]:


import kaldo.controllers.plotter as plotter
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8')

# Plot dispersion relation and group velocity in each direction
plotter.plot_dispersion(phonons,n_k_points=int(k_label))

