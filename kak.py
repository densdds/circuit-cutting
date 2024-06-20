"""KAK decomposition of two-qubit unitary"""

import numpy as np 
import scipy as sp
import functools

from typing import Union, TYPE_CHECKING
if TYPE_CHECKING:
    from numpy.typing import DTypeLike, ArrayLike

from typing import Tuple, Callable, List, Optional

from conditions import *

X = np.array([[0, 1],
              [1, 0]])

Y = np.array([[0, -1j],
              [1j, 0]])

Z = np.array([[1, 0],
              [0, -1]])

XX = np.kron(X, X)
YY = np.kron(Y, Y)
ZZ = np.kron(Z, Z)

MAGIC = 1/np.sqrt(2) * np.array([[1, 0, 0, 1j],
                                [0, 1j, 1, 0],
                                [0, 1j, -1, 0],
                                [1, 0, 0, -1j]])

MAGIC_DAG = np.conjugate(np.transpose(MAGIC))

GAMMA = np.array([[1, 1, 1, 1],
                  [1, 1, -1, -1],
                  [-1, 1, -1, 1],
                  [1, -1, -1, 1]]) * 0.25

def unitary_eig(
    matrix: np.ndarray, check_preconditions: bool = True, atol: float = 1e-8
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Gives the guaranteed unitary eigendecomposition of a normal matrix.

    All hermitian and unitary matrices are normal matrices. This method was
    introduced as for certain classes of unitary matrices (where the eigenvalues
    are close to each other) the eigenvectors returned by `numpy.linalg.eig` are
    not guaranteed to be orthogonal.
    For more information, see https://github.com/numpy/numpy/issues/15461.

    Args:
        matrix: A normal matrix. If not normal, this method is not
            guaranteed to return correct eigenvalues.  A normal matrix
            is one where $A A^\dagger = A^\dagger A$.
        check_preconditions: When true and matrix is not unitary,
            a `ValueError` is raised when the matrix is not normal.
        atol: The absolute tolerance when checking whether the original matrix
            was unitary.

    Returns:
        A Tuple of
            eigvals: The eigenvalues of `matrix`.
            V: The unitary matrix with the eigenvectors as columns.

    Raises:
        ValueError: if the input matrix is not normal.
    """
    if check_preconditions and not is_normal(matrix, atol=atol):
        raise ValueError(f'Input must correspond to a normal matrix .Received input:\n{matrix}')

    R, V = sp.linalg.schur(matrix, output="complex")
    return R.diagonal(), V

def map_eigenvalues(
    matrix: np.ndarray, func: Callable[[complex], complex], *, atol: float = 1e-8
) -> np.ndarray:
    """Applies a function to the eigenvalues of a matrix.

    Given M = sum_k a_k |v_k><v_k|, returns f(M) = sum_k f(a_k) |v_k><v_k|.

    Args:
        matrix: The matrix to modify with the function.
        func: The function to apply to the eigenvalues of the matrix.
        rtol: Relative threshold used when separating eigenspaces.
        atol: Absolute threshold used when separating eigenspaces.

    Returns:
        The transformed matrix.
    """
    vals, vecs = unitary_eig(matrix, atol=atol)
    pieces = [np.outer(vec, np.conj(vec.T)) for vec in vecs.T]
    out_vals = np.vectorize(func)(vals.astype(complex))

    total = np.zeros(shape=matrix.shape)
    for piece, val in zip(pieces, out_vals):
        total = np.add(total, piece * val)
    return total


def kron_factor_4x4_to_2x2s(matrix: np.ndarray) -> Tuple[complex, np.ndarray, np.ndarray]:
    """Splits a 4x4 matrix U = kron(A, B) into A, B, and a global factor.

    Requires the matrix to be the kronecker product of two 2x2 unitaries.
    Requires the matrix to have a non-zero determinant.
    Giving an incorrect matrix will cause garbage output.

    Args:
        matrix: The 4x4 unitary matrix to factor.

    Returns:
        A scalar factor and a pair of 2x2 unit-determinant matrices. The
        kronecker product of all three is equal to the given matrix.

    Raises:
        ValueError:
            The given matrix can't be tensor-factored into 2x2 pieces.
    """

    # Use the entry with the largest magnitude as a reference point.
    a, b = max(((i, j) for i in range(4) for j in range(4)), key=lambda t: abs(matrix[t]))

    # Extract sub-factors touching the reference cell.
    f1 = np.zeros((2, 2), dtype=np.complex128)
    f2 = np.zeros((2, 2), dtype=np.complex128)
    for i in range(2):
        for j in range(2):
            f1[(a >> 1) ^ i, (b >> 1) ^ j] = matrix[a ^ (i << 1), b ^ (j << 1)]
            f2[(a & 1) ^ i, (b & 1) ^ j] = matrix[a ^ i, b ^ j]

    # Rescale factors to have unit determinants.
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 /= np.sqrt(np.linalg.det(f1)) or 1
        f2 /= np.sqrt(np.linalg.det(f2)) or 1

    # Determine global phase.
    g = matrix[a, b] / (f1[a >> 1, b >> 1] * f2[a & 1, b & 1])
    if np.real(g) < 0:
        f1 *= -1
        g = -g

    return g, f1, f2

def so4_to_magic_su2s(
    mat: np.ndarray, *, rtol: float = 1e-5, atol: float = 1e-8, check_preconditions: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """Finds 2x2 special-unitaries A, B where mat = Mag.H @ kron(A, B) @ Mag.

    Mag is the magic basis matrix:

        1  0  0  i
        0  i  1  0
        0  i -1  0     (times sqrt(0.5) to normalize)
        1  0  0 -i

    Args:
        mat: A real 4x4 orthogonal matrix.
        rtol: Per-matrix-entry relative tolerance on equality.
        atol: Per-matrix-entry absolute tolerance on equality.
        check_preconditions: When set, the code verifies that the given
            matrix is from SO(4). Defaults to set.

    Returns:
        A pair (A, B) of matrices in SU(2) such that Mag.H @ kron(A, B) @ Mag
        is approximately equal to the given matrix.

    Raises:
        ValueError: Bad matrix.
    """
    if check_preconditions:
        if mat.shape != (4, 4) or not is_special_orthogonal(mat, atol=atol, rtol=rtol):
            raise ValueError('mat must be 4x4 special orthogonal.')

    ab = np.linalg.multi_dot([MAGIC, mat, MAGIC_DAG])
    _, a, b = kron_factor_4x4_to_2x2s(ab)

    return a, b


def merge_dtypes(dtype1: 'DTypeLike', dtype2: 'DTypeLike') -> np.dtype:
    return (np.zeros(0, dtype1) + np.zeros(0, dtype2)).dtype

def contiguous_groups(
    length: int, comparator: Callable[[int, int], bool]
) -> List[Tuple[int, int]]:
    """Splits range(length) into approximate equivalence classes.

    Args:
        length: The length of the range to split.
        comparator: Determines if two indices have approximately equal items.

    Returns:
        A list of (inclusive_start, exclusive_end) range endpoints. Each
        corresponds to a run of approximately-equivalent items.
    """
    result = []
    start = 0
    while start < length:
        past = start + 1
        while past < length and comparator(start, past):
            past += 1
        result.append((start, past))
        start = past
    return result

def block_diag(*blocks: np.ndarray) -> np.ndarray:
    """Concatenates blocks into a block diagonal matrix.

    Args:
        *blocks: Square matrices to place along the diagonal of the result.

    Returns:
        A block diagonal matrix with the given blocks along its diagonal.

    Raises:
        ValueError: A block isn't square.
    """
    for b in blocks:
        if b.shape[0] != b.shape[1]:
            raise ValueError('Blocks must be square.')

    if not blocks:
        return np.zeros((0, 0), dtype=np.complex128)

    n = sum(b.shape[0] for b in blocks)
    dtype = functools.reduce(merge_dtypes, (b.dtype for b in blocks))

    result = np.zeros(shape=(n, n), dtype=dtype)
    i = 0
    for b in blocks:
        j = i + b.shape[0]
        result[i:j, i:j] = b
        i = j

    return result

def svd_handling_empty(mat):
    if not mat.shape[0] * mat.shape[1]:
        z = np.zeros((0, 0), dtype=mat.dtype)
        return z, np.array([]), z

    return np.linalg.svd(mat)

def diagonalize_real_symmetric_matrix(
    matrix: np.ndarray, *, rtol: float = 1e-5, atol: float = 1e-8, check_preconditions: bool = True
) -> np.ndarray:
    """Returns an orthogonal matrix that diagonalizes the given matrix.

    Args:
        matrix: A real symmetric matrix to diagonalize.
        rtol: Relative error tolerance.
        atol: Absolute error tolerance.
        check_preconditions: If set, verifies that the input matrix is real and
            symmetric.

    Returns:
        An orthogonal matrix P such that P.T @ matrix @ P is diagonal.

    Raises:
        ValueError: Matrix isn't real symmetric.
    """

    if check_preconditions and (
        np.any(np.imag(matrix) != 0) or not is_hermitian(matrix, rtol=rtol, atol=atol)
    ):
        raise ValueError('Input must be real and symmetric.')

    _, result = np.linalg.eigh(matrix)

    return result

def diagonalize_real_symmetric_and_sorted_diagonal_matrices(
    symmetric_matrix: np.ndarray,
    diagonal_matrix: np.ndarray,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    check_preconditions: bool = True,
) -> np.ndarray:
    """Returns an orthogonal matrix that diagonalizes both given matrices.

    The given matrices must commute.
    Guarantees that the sorted diagonal matrix is not permuted by the
    diagonalization (except for nearly-equal values).

    Args:
        symmetric_matrix: A real symmetric matrix.
        diagonal_matrix: A real diagonal matrix with entries along the diagonal
            sorted into descending order.
        rtol: Relative numeric error threshold.
        atol: Absolute numeric error threshold.
        check_preconditions: If set, verifies that the input matrices commute
            and are respectively symmetric and diagonal descending.

    Returns:
        An orthogonal matrix P such that P.T @ symmetric_matrix @ P is diagonal
        and P.T @ diagonal_matrix @ P = diagonal_matrix (up to tolerance).

    Raises:
        ValueError: Matrices don't meet preconditions (e.g. not symmetric).
    """

    # Verify preconditions.
    if check_preconditions:
        if np.any(np.imag(symmetric_matrix)) or not is_hermitian(
            symmetric_matrix, rtol=rtol, atol=atol
        ):
            raise ValueError('symmetric_matrix must be real symmetric.')
        if (
            not is_diagonal(diagonal_matrix, atol=atol)
            or np.any(np.imag(diagonal_matrix))
            or np.any(diagonal_matrix[:-1, :-1] < diagonal_matrix[1:, 1:])
        ):
            raise ValueError('diagonal_matrix must be real diagonal descending.')
        if not matrix_commutes(diagonal_matrix, symmetric_matrix, rtol=rtol, atol=atol):
            raise ValueError('Given matrices must commute.')

    def similar_singular(i, j):
        return np.allclose(diagonal_matrix[i, i], diagonal_matrix[j, j], rtol=rtol)

    # Because the symmetric matrix commutes with the diagonal singulars matrix,
    # the symmetric matrix should be block-diagonal with a block boundary
    # wherever the singular values happen change. So we can use the singular
    # values to extract blocks that can be independently diagonalized.
    ranges = contiguous_groups(diagonal_matrix.shape[0], similar_singular)

    # Build the overall diagonalization by diagonalizing each block.
    p = np.zeros(symmetric_matrix.shape, dtype=np.float64)
    for start, end in ranges:
        block = symmetric_matrix[start:end, start:end]
        p[start:end, start:end] = diagonalize_real_symmetric_matrix(
            block, rtol=rtol, atol=atol, check_preconditions=False
        )

    return p

def diag_real_matrix_pair_with_symmetric_products(
    mat1: np.ndarray,
    mat2: np.ndarray,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    check_preconditions: bool = True,
):
    """Finds orthogonal matrices that diagonalize both mat1 and mat2.

    Requires mat1 and mat2 to be real.
    Requires mat1.T @ mat2 to be symmetric.
    Requires mat1 @ mat2.T to be symmetric.

    Args:
        mat1: One of the real matrices.
        mat2: The other real matrix.
        rtol: Relative numeric error threshold.
        atol: Absolute numeric error threshold.
        check_preconditions: If set, verifies that the inputs are real, and that
            mat1.T @ mat2 and mat1 @ mat2.T are both symmetric. Defaults to set.

    Returns:
        A tuple (L, R) of two orthogonal matrices, such that both L @ mat1 @ R
        and L @ mat2 @ R are diagonal matrices.

    Raises:
        ValueError: Matrices don't meet preconditions (e.g. not real).
    """

    if check_preconditions:
        if np.any(np.imag(mat1) != 0):
            raise ValueError('mat1 must be real.')
        if np.any(np.imag(mat2) != 0):
            raise ValueError('mat2 must be real.')
        if not is_hermitian(np.dot(mat1, mat2.T), rtol=rtol, atol=atol):
            raise ValueError('mat1 @ mat2.T must be symmetric.')
        if not is_hermitian(np.dot(mat1.T, mat2), rtol=rtol, atol=atol):
            raise ValueError('mat1.T @ mat2 must be symmetric.')

    # Use SVD to bi-diagonalize the first matrix.
    base_left, base_diag, base_right = svd_handling_empty(np.real(mat1))
    base_diag = np.diag(base_diag)

    # Determine where we switch between diagonalization-fixup strategies.
    dim = base_diag.shape[0]
    rank = dim
    while rank > 0 and all_near_zero(base_diag[rank - 1, rank - 1], atol=atol):
        rank -= 1
    base_diag = base_diag[:rank, :rank]

    # Try diagonalizing the second matrix with the same factors as the first.
    semi_corrected = np.linalg.multi_dot([base_left.T, np.real(mat2), base_right.T])

    # Fix up the part of the second matrix's diagonalization that's matched
    # against non-zero diagonal entries in the first matrix's diagonalization
    # by performing simultaneous diagonalization.
    overlap = semi_corrected[:rank, :rank]
    overlap_adjust = diagonalize_real_symmetric_and_sorted_diagonal_matrices(
        overlap, base_diag, rtol=rtol, atol=atol, check_preconditions=check_preconditions
    )

    # Fix up the part of the second matrix's diagonalization that's matched
    # against zeros in the first matrix's diagonalization by performing an SVD.
    extra = semi_corrected[rank:, rank:]
    extra_left_adjust, _, extra_right_adjust = svd_handling_empty(extra)

    # Merge the fixup factors into the initial diagonalization.
    left_adjust = block_diag(overlap_adjust, extra_left_adjust)
    right_adjust = block_diag(overlap_adjust.T, extra_right_adjust)
    left = np.dot(left_adjust.T, base_left.T)
    right = np.dot(base_right.T, right_adjust.T)

    return left, right

def diag_unitary_with_special_orthogonals(
    mat: np.ndarray, *, rtol: float = 1e-5, atol: float = 1e-8, check_preconditions: bool = True
):
    """Finds orthogonal matrices L, R such that L @ matrix @ R is diagonal.

    Args:
        mat: A unitary matrix.
        rtol: Relative numeric error threshold.
        atol: Absolute numeric error threshold.
        check_preconditions: If set, verifies that the input is a unitary matrix
            (to the given tolerances). Defaults to set.

    Returns:
        A triplet (L, d, R) such that L @ mat @ R = diag(d). Both L and R will
        be orthogonal matrices with determinant equal to 1.

    Raises:
        ValueError: Matrices don't meet preconditions (e.g. not real).
    """

    if check_preconditions:
        if not np.allclose(mat @ mat.T.conj(), np.eye(mat.shape[0])):
            raise ValueError('matrix must be unitary.')

    # Note: Because mat is unitary, setting A = real(mat) and B = imag(mat)
    # guarantees that both A @ B.T and A.T @ B are Hermitian.
    left, right = diag_real_matrix_pair_with_symmetric_products(
        np.real(mat), np.imag(mat), rtol=rtol, atol=atol, check_preconditions=check_preconditions
    )

    # Convert to special orthogonal w/o breaking diagonalization.
    with np.errstate(divide="ignore", invalid="ignore"):
        if np.linalg.det(left) < 0:
            left[0, :] *= -1
        if np.linalg.det(right) < 0:
            right[:, 0] *= -1

    diag = np.linalg.multi_dot([left, mat, right])

    return left, np.diag(diag), right

class KakDecomposition:
    """A convenient description of an arbitrary two-qubit operation.

    Any two qubit operation U can be decomposed into the form

        U = g · (a1 ⊗ a0) · exp(i·(x·XX + y·YY + z·ZZ)) · (b1 ⊗ b0)

    This class stores g, (b0, b1), (x, y, z), and (a0, a1).

    Attributes:
        global_phase: g from the above equation.
        single_qubit_operations_before: b0, b1 from the above equation.
        interaction_coefficients: x, y, z from the above equation.
        single_qubit_operations_after: a0, a1 from the above equation.
    """
    def __init__(
        self,
        *,
        global_phase: complex = complex(1),
        single_qubit_operations_before: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        interaction_coefficients: Tuple[float, float, float],
        single_qubit_operations_after: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> None:
        """Initializes a decomposition for a two-qubit operation U.

        U = g · (a1 ⊗ a0) · exp(i·(x·XX + y·YY + z·ZZ)) · (b1 ⊗ b0)

        Args:
            global_phase: g from the above equation.
            single_qubit_operations_before: b0, b1 from the above equation.
            interaction_coefficients: x, y, z from the above equation.
            single_qubit_operations_after: a0, a1 from the above equation.
        """
        self.global_phase: complex = global_phase
        self.single_qubit_operations_before: Tuple[np.ndarray, np.ndarray] = (
            single_qubit_operations_before
            or (np.eye(2, dtype=np.complex64), np.eye(2, dtype=np.complex64))
        )
        self.interaction_coefficients = interaction_coefficients
        self.single_qubit_operations_after: Tuple[np.ndarray, np.ndarray] = (
            single_qubit_operations_after
            or (np.eye(2, dtype=np.complex64), np.eye(2, dtype=np.complex64))
        )

    def _unitary_(self) -> np.ndarray:
        """Returns the decomposition's two-qubit unitary matrix.

        U = g · (a1 ⊗ a0) · exp(i·(x·XX + y·YY + z·ZZ)) · (b1 ⊗ b0)
        """
        before = np.kron(*self.single_qubit_operations_before)
        after = np.kron(*self.single_qubit_operations_after)

        def interaction_matrix(m: np.ndarray, c: float) -> np.ndarray:
            return map_eigenvalues(np.kron(m, m), lambda v: np.exp(1j * v * c))

        x, y, z = self.interaction_coefficients
        x_mat = np.array([[0, 1], [1, 0]])
        y_mat = np.array([[0, -1j], [1j, 0]])
        z_mat = np.array([[1, 0], [0, -1]])

        return self.global_phase * np.linalg.multi_dot([
            after,
            interaction_matrix(z_mat, z),
            interaction_matrix(y_mat, y),
            interaction_matrix(x_mat, x),
            before,
        ])
    
    
def kak_canonicalize_vector(x: float, y: float, z: float, atol: float = 1e-9) -> KakDecomposition:
    """Canonicalizes an XX/YY/ZZ interaction by swap/negate/shift-ing axes.

    Args:
        x: The strength of the XX interaction.
        y: The strength of the YY interaction.
        z: The strength of the ZZ interaction.
        atol: How close x2 must be to π/4 to guarantee z2 >= 0

    Returns:
        The canonicalized decomposition, with vector coefficients (x2, y2, z2)
        satisfying:

            0 ≤ abs(z2) ≤ y2 ≤ x2 ≤ π/4
            if x2 = π/4, z2 >= 0

        Guarantees that the implied output matrix:

            g · (a1 ⊗ a0) · exp(i·(x2·XX + y2·YY + z2·ZZ)) · (b1 ⊗ b0)

        is approximately equal to the implied input matrix:

            exp(i·(x·XX + y·YY + z·ZZ))
    """

    phase = [complex(1)]  # Accumulated global phase.
    left = [np.eye(2)] * 2  # Per-qubit left factors.
    right = [np.eye(2)] * 2  # Per-qubit right factors.
    v = [x, y, z]  # Remaining XX/YY/ZZ interaction vector.

    # These special-unitary matrices flip the X, Y, and Z axes respectively.
    flippers = [
        np.array([[0, 1], [1, 0]]) * 1j,
        np.array([[0, -1j], [1j, 0]]) * 1j,
        np.array([[1, 0], [0, -1]]) * 1j,
    ]

    # Each of these special-unitary matrices swaps two the roles of two axes.
    # The matrix at index k swaps the *other two* axes (e.g. swappers[1] is a
    # Hadamard operation that swaps X and Z).
    swappers = [
        np.array([[1, -1j], [1j, -1]]) * 1j * np.sqrt(0.5),
        np.array([[1, 1], [1, -1]]) * 1j * np.sqrt(0.5),
        np.array([[0, 1 - 1j], [1 + 1j, 0]]) * 1j * np.sqrt(0.5),
    ]

    # Shifting strength by ½π is equivalent to local ops (e.g. exp(i½π XX)∝XX).
    def shift(k, step):
        v[k] += step * np.pi / 2
        phase[0] *= 1j**step
        right[0] = np.linalg.multi_dot([flippers[k] ** (step % 4), right[0]])
        right[1] = np.linalg.multi_dot([flippers[k] ** (step % 4), right[1]])

    # Two negations is equivalent to temporarily flipping along the other axis.
    def negate(k1, k2):
        v[k1] *= -1
        v[k2] *= -1
        phase[0] *= -1
        s = flippers[3 - k1 - k2]  # The other axis' flipper.
        left[1] = np.linalg.multi_dot([left[1], s])
        right[1] = np.linalg.multi_dot([s, right[1]])

    # Swapping components is equivalent to temporarily swapping the two axes.
    def swap(k1, k2):
        v[k1], v[k2] = v[k2], v[k1]
        s = swappers[3 - k1 - k2]  # The other axis' swapper.
        left[0] = np.linalg.multi_dot([left[0], s])
        left[1] = np.linalg.multi_dot([left[1], s])
        right[0] = np.linalg.multi_dot([s, right[0]])
        right[1] = np.linalg.multi_dot([s, right[1]])

    # Shifts an axis strength into the range (-π/4, π/4].
    def canonical_shift(k):
        while v[k] <= -np.pi / 4:
            shift(k, +1)
        while v[k] > np.pi / 4:
            shift(k, -1)

    # Sorts axis strengths into descending order by absolute magnitude.
    def sort():
        if abs(v[0]) < abs(v[1]):
            swap(0, 1)
        if abs(v[1]) < abs(v[2]):
            swap(1, 2)
        if abs(v[0]) < abs(v[1]):
            swap(0, 1)

    # Get all strengths to (-¼π, ¼π] in descending order by absolute magnitude.
    canonical_shift(0)
    canonical_shift(1)
    canonical_shift(2)
    sort()

    # Move all negativity into z.
    if v[0] < 0:
        negate(0, 2)
    if v[1] < 0:
        negate(1, 2)
    canonical_shift(2)

    # If x = π/4, force z to be positive
    if v[0] > np.pi / 4 - atol and v[2] < 0:
        shift(0, -1)
        negate(0, 2)

    return KakDecomposition(
        global_phase=phase[0],
        single_qubit_operations_after=(left[1], left[0]),
        interaction_coefficients=(v[0], v[1], v[2]),
        single_qubit_operations_before=(right[1], right[0]),
    )

def kak_decomposition(
    unitary_object: Union[
        np.ndarray, KakDecomposition
    ],
    *,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    check_preconditions: bool = True,
) -> KakDecomposition:
    """Decomposes a 2-qubit unitary into 1-qubit ops and XX/YY/ZZ interactions.

    Args:
        unitary_object: The value to decompose. Can either be a 4x4 unitary
            matrix, or an object that has a 4x4 unitary matrix (via the
            `cirq.SupportsUnitary` protocol).
        rtol: Per-matrix-entry relative tolerance on equality.
        atol: Per-matrix-entry absolute tolerance on equality.
        check_preconditions: If set, verifies that the input corresponds to a
            4x4 unitary before decomposing.

    Returns:
        A `cirq.KakDecomposition` canonicalized such that the interaction
        coefficients x, y, z satisfy:

            0 ≤ abs(z2) ≤ y2 ≤ x2 ≤ π/4
            if x2 = π/4, z2 >= 0

    Raises:
        ValueError: Bad matrix.
        ArithmeticError: Failed to perform the decomposition.
    """

    if isinstance(unitary_object, KakDecomposition):
        return unitary_object
    if isinstance(unitary_object, np.ndarray):
        mat = unitary_object
    if check_preconditions and (
        mat.shape != (4, 4) or not is_unitary(mat, rtol=rtol, atol=atol)
    ):
        raise ValueError(f'Input must correspond to a 4x4 unitary matrix. Received matrix:\n{mat}')

    # Diagonalize in magic basis.
    left, d, right = diag_unitary_with_special_orthogonals(
        MAGIC_DAG @ mat @ MAGIC, atol=atol, rtol=rtol, check_preconditions=False
    )

    # Recover pieces.
    a1, a0 = so4_to_magic_su2s(left.T, atol=atol, rtol=rtol, check_preconditions=False)
    b1, b0 = so4_to_magic_su2s(right.T, atol=atol, rtol=rtol, check_preconditions=False)
    w, x, y, z = (GAMMA @ np.angle(d).reshape(-1, 1)).flatten()
    g = np.exp(1j * w)

    # Canonicalize.
    inner_cannon = kak_canonicalize_vector(x, y, z)

    b1 = np.dot(inner_cannon.single_qubit_operations_before[0], b1)
    b0 = np.dot(inner_cannon.single_qubit_operations_before[1], b0)
    a1 = np.dot(a1, inner_cannon.single_qubit_operations_after[0])
    a0 = np.dot(a0, inner_cannon.single_qubit_operations_after[1])

    return KakDecomposition(
        interaction_coefficients=inner_cannon.interaction_coefficients,
        global_phase=g * inner_cannon.global_phase,
        single_qubit_operations_before=(b1, b0),
        single_qubit_operations_after=(a1, a0),
    )
