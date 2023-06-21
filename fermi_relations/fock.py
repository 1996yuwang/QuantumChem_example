import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from fermionic_operators import orbital_create_op


def slater_determinant(phi):
    """
    Create the Slater determinant of single-particle states stored as column vectors in `phi`.
    """
    phi = np.asarray(phi)
    nmodes = phi.shape[0]
    # vacuum state
    psi = np.zeros(2**nmodes)
    psi[0] = 1
    for i in range(phi.shape[1]):
        psi = orbital_create_op(phi[:, i]) @ psi
    return psi


def fock_orbital_base_change(U):
    """
    Construct the matrix representation of a unitary, single-particle
    base change matrix described by `U` on the full Fock space.
    """
    U = np.asarray(U)
    nmodes = U.shape[1]
    clist = [orbital_create_op(U[:, i]) for i in range(nmodes)]
    UF = lil_matrix((2**nmodes, 2**nmodes), dtype=U.dtype)
    for m in range(2**nmodes):
        # vacuum state
        psi = np.zeros(2**nmodes)
        psi[0] = 1
        for i in range(nmodes):
            if m & (1 << (nmodes - i - 1)):
                psi = clist[i] @ psi
        UF[m] = psi
    return csr_matrix(UF.T)
