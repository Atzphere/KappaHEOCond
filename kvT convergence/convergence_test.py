from ase.build import bulk
from ase.calculators.lammpslib import LAMMPSlib
from kaldo.forceconstants import ForceConstants
import numpy as np

import multiprocess as mp

# We start from the atoms object. use ASE to initialize a silicon diamond unit cell with FCC lattice parameter a.
atoms = bulk('Si', 'diamond', a=5.432)

# Config super cell and calculator input
# use a 5x5x5 supercell of unit cells.
# can also specify a supercell parameter exclusively for third-order force constants.
# additional supercells along for computing higher-order nearest-neighor terms.
supercell = np.array([5, 5, 5])
lammps_inputs = {
    'lmpcmds': [
        'pair_style tersoff',
        'pair_coeff * * Si.tersoff Si'],

    'log_file': 'lammps-si-bulk.log',
    'keep_alive': True}

# Create a finite difference object
forceconstants_config = {'atoms': atoms,
                         'supercell': supercell, 'folder': 'fd'}
forceconstants = ForceConstants(**forceconstants_config)

# Compute 2nd and 3rd IFCs with LAMMPS using a tersoff potential
# delta_shift is how much to move atoms when computing forces
forceconstants.second.calculate(LAMMPSlib(**lammps_inputs), delta_shift=1e-3)
forceconstants.third.calculate(LAMMPSlib(**lammps_inputs), delta_shift=1e-3)


from kaldo.conductivity import Conductivity
from kaldo.phonons import Phonons
import pandas as pd

import logging
logging.basicConfig(level=logging.INFO)  # Set the root logger to WARNING level

temperatures = np.linspace(200, 1000, 9)  # K
K = np.arange(6, 20, 2)


def temp_series(k):
    logging.basicConfig(level=logging.WARNING)
    '''
    quick and dirty function for convergence testing 
    generates kappa vs t data for a variety of methods
    sampling k-space with a (k,k,k) grid in the BZ.

    note - this will only work optimally for cubic lattices
    '''
    kappa_qhgk = []
    kappa_rtaq = []
    kappa_invq = []

    for T in temperatures:
        # Define k-point grids, temperature
        # and the assumption for the
        # phonon poluation (i.e classical vs. quantum)

        kpts = [k, k, k]
        temperature = 300
        is_classic = False  # phonon mondes are treated as quantized
        k_label = str(k) + '_' + str(k) + '_' + str(k)

        # Create a phonon object
        phonons = Phonons(forceconstants=forceconstants,
                          kpts=kpts,
                          is_classic=is_classic,
                          temperature=T,
                          folder='si-bulk-ald-' + k_label,
                          storage='numpy')

        # compute conductivity with full scattering matrix inversion
        print('\n')
        inv_cond_matrix = (Conductivity(
            phonons=phonons, method='inverse').conductivity.sum(axis=0))
        cond_inv = float(np.mean(np.diag(inv_cond_matrix)))
        print('Inverted conductivity (W/mK): %.3f' % (cond_inv))

        # Calculate conductivity  with  relaxation time approximation (rta), only considering diagonals of scattering matrix.
        # this effectively means we only consider energy -conserving three-phonon processes where the two initial (annihilation to 1 phonon)
        # or two final (one phonon decays into two) phonons have the momentum index and mode.
        print('\n')
        rta_cond_matrix = Conductivity(
            phonons=phonons, method='rta', n_interations=0).conductivity.sum(axis=0)
        cond_rta = float(np.mean(np.diag(rta_cond_matrix)))
        print('RTA conductivity (W/mK): %.3f' % (cond_rta))

        qhgk_cond_matrix = Conductivity(
            phonons=phonons, method='qhgk').conductivity.sum(axis=0)
        cond_qhgk = float(np.mean(np.diag(qhgk_cond_matrix)))

        print(f'QHGK conductivity (W/mK): {cond_qhgk:.3f}')

        kappa_qhgk.append(cond_qhgk)
        kappa_rtaq.append(cond_rta)
        kappa_invq.append(cond_inv)

    df = pd.DataFrame(np.array([kappa_qhgk, kappa_rtaq, kappa_invq]).T, columns=[f"qhgk k={k}", f"rtaq k={k}", f"inv k={k}"])
    df.to_csv(f'./sets/out_{k}.csv')
    return df


if __name__ == "__main__":
    print("Starting worker processes")
    with mp.Pool(mp.cpu_count()) as p:
        res = p.map(temp_series, K)
