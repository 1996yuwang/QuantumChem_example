import unittest
import numpy as np
from scipy.linalg import expm
from scipy.stats import unitary_group
import scipy.sparse.linalg as spla
from scipy import sparse
from fermionic_operators import construct_fermionic_operators, orbital_number_op, total_number_op
from fock import slater_determinant, fock_orbital_base_change
from util import crandn


class TestFock(unittest.TestCase):

    def test_total_number_op(self):
        """
        Test construction of the total number operator.
        """
        for nmodes in range(1, 8):
            _, _, nlist = construct_fermionic_operators(nmodes)
            ntot = total_number_op(nmodes)
            self.assertEqual(spla.norm(ntot - sum(nlist)), 0)

    def test_slater_determinant(self):
        """
        Test Slater determinant construction by simulating
        quantum time evolution of a Slater determinant in two alternative ways.
        """
        rng = np.random.default_rng()

        # number of modes
        nmodes = 7
        # number of particles
        N = 3
        # random orthonormal states
        base = unitary_group.rvs(nmodes, random_state=rng)
        orb = base[:, :N]

        # create Slater determinant
        psi = slater_determinant(orb)
        # must be normalized
        self.assertAlmostEqual(np.linalg.norm(psi), 1, delta=1e-13)
        # must be eigenstate of number operator
        self.assertAlmostEqual(np.linalg.norm(
            total_number_op(nmodes) @ psi - N*psi), 0, delta=1e-13)
        for i in range(nmodes):
            # number operator of an individual mode
            n = orbital_number_op(base[:, i])
            self.assertAlmostEqual(np.linalg.norm(
                n @ psi - (1 if i < N else 0) * psi), 0, delta=1e-13)

        # random single-particle Hamiltonian
        h = crandn((nmodes, nmodes), rng)
        h = 0.5*(h + h.conj().T)
        # Hamiltonian on full Fock space
        clist, alist, _ = construct_fermionic_operators(nmodes)
        H = sum(h[i, j] * (clist[i] @ alist[j]) for i in range(nmodes) for j in range(nmodes))

        # energy expectation value
        en = np.vdot(psi, H.toarray() @ psi)
        self.assertAlmostEqual(en, np.trace(orb.conj().T @ h @ orb))

        # time-evolved state
        psi_t = expm(-1j*H.toarray()) @ psi
        # alternative construction: time-evolve single-particle states individually
        orb_t = expm(-1j*h) @ orb
        psi_t_alt = slater_determinant(orb_t)
        # compare
        self.assertAlmostEqual(np.linalg.norm(psi_t_alt - psi_t), 0, delta=1e-13)

    def test_orbital_base_change(self):
        """
        Test matrix representation of single-particle base change
        on overall Fock space.
        """
        rng = np.random.default_rng()

        # number of modes
        nmodes = 5

        # for a single-particle identity map, the overall base change matrix
        # should likewise be the identity map
        UF = fock_orbital_base_change(np.identity(nmodes))
        self.assertEqual(spla.norm(UF - sparse.identity(2**nmodes)), 0)

        # random orthonormal states
        U = unitary_group.rvs(nmodes, random_state=rng)
        self.assertAlmostEqual(np.linalg.norm(U.conj().T @ U - np.identity(nmodes)), 0, delta=1e-13)

        UF = fock_orbital_base_change(U)
        # must likewise be unitary
        self.assertAlmostEqual(spla.norm(UF.conj().T @ UF - sparse.identity(2**nmodes)), 0, delta=1e-13)

        idx = [0, 2, 3]
        psi_ref = slater_determinant(U[:, idx])
        # encode indices in binary format
        i = sum(1 << (nmodes - j - 1) for j in idx)
        # need to reshape since slicing returns matrix (different from numpy convention)
        psi = np.reshape(UF[:, i].toarray(), -1)
        # compare
        self.assertTrue(np.allclose(psi, psi_ref))

    def test_thouless_theorem(self):
        """
        Numerically verify Thouless' theorem.
        """
        rng = np.random.default_rng()

        # number of modes
        nmodes = 5

        # random single-particle base change matrix as matrix exponential
        h = crandn((nmodes, nmodes), rng)
        # identity even holds if 'h' is not Hermitian (and 'U' not unitary)
        # h = 0.5*(h + h.conj().T)
        U = expm(-1j*h)

        clist, alist, _ = construct_fermionic_operators(nmodes)
        T = sum(h[i, j] * (clist[i] @ alist[j]) for i in range(nmodes) for j in range(nmodes))
        UF = expm(-1j*T.toarray())

        # reference base change matrix on full Fock space
        UF_ref = fock_orbital_base_change(U)

        # compare
        self.assertTrue(np.allclose(UF, UF_ref.toarray()))

    def test_skew_number_op(self):
        """
        Test action of a number operator w.r.t. a non-orthogonal basis state on a Slater determinant.
        """
        rng = np.random.default_rng()

        # number of modes
        nmodes = 7
        # number of particles
        N = 3
        # random orthonormal states
        base = unitary_group.rvs(nmodes, random_state=rng)
        orb = base[:, :N]

        # create Slater determinant
        psi = slater_determinant(orb)

        # non-orthogonal state
        x = crandn(nmodes, rng)
        x /= np.linalg.norm(x)
        n = orbital_number_op(x)

        # manually construct sum of Slater determinants by projecting one orbital at a time onto 'x'
        n_psi = 0
        for i in range(N):
            n_orbs_i = np.concatenate((orb[:, :i],
                                       np.reshape(np.vdot(x, orb[:, i]) * x, (nmodes, 1)),
                                       orb[:, i+1:]), axis=1)
            n_psi += slater_determinant(n_orbs_i)
        # compare
        self.assertTrue(np.allclose(n_psi, n @ psi))


if __name__ == '__main__':
    unittest.main()
