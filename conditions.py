"""Functions for checking different conditions for matrices"""

import numpy as np

def all_near_zero(a, *, atol: float = 1e-8) -> bool:
    """Checks if the tensor's elements are all near zero.

    Args:
        a: Tensor of elements that could all be near zero.
        atol: Absolute tolerance.
    """
    return bool(np.all(np.less_equal(np.abs(a), atol)))

def is_hermitian(mat, rtol, atol) -> bool:
    """Cheks if the matrix is hermitian"""
    return np.allclose(mat, mat.T.conj(), rtol, atol)

def is_unitary(mat, rtol, atol) -> bool:
    """Checks if the matrix is unitary"""
    return np.allclose(mat @ mat.T.conj(), np.eye(mat.shape[0]), rtol, atol)

def is_diagonal(matrix: np.ndarray, *, atol: float = 1e-8) -> bool:
    """Determines if a matrix is a approximately diagonal.

    A matrix is diagonal if i!=j implies m[i,j]==0.

    Args:
        matrix: The matrix to check.
        atol: The per-matrix-entry absolute tolerance on equality.

    Returns:
        Whether the matrix is diagonal within the given tolerance.
    """
    matrix = np.copy(matrix)
    for i in range(min(matrix.shape)):
        matrix[i, i] = 0
    return all_near_zero(matrix, atol=atol)

def is_orthogonal(matrix: np.ndarray, *, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
    """Determines if a matrix is approximately orthogonal.

    A matrix is orthogonal if it's square and real and its transpose is its
    inverse.

    Args:
        matrix: The matrix to check.
        rtol: The per-matrix-entry relative tolerance on equality.
        atol: The per-matrix-entry absolute tolerance on equality.

    Returns:
        Whether the matrix is orthogonal within the given tolerance.
    """
    return (
        matrix.shape[0] == matrix.shape[1]
        and np.all(np.imag(matrix) == 0).item()
        and np.allclose(matrix.dot(matrix.T), np.eye(matrix.shape[0]), rtol=rtol, atol=atol)
    )

def is_special_orthogonal(matrix: np.ndarray, *, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
    """Determines if a matrix is approximately special orthogonal.

    A matrix is special orthogonal if it is square and real and its transpose
    is its inverse and its determinant is one.

    Args:
        matrix: The matrix to check.
        rtol: The per-matrix-entry relative tolerance on equality.
        atol: The per-matrix-entry absolute tolerance on equality.

    Returns:
        Whether the matrix is special orthogonal within the given tolerance.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        return is_orthogonal(matrix, rtol=rtol, atol=atol) and (
            matrix.shape[0] == 0 or np.allclose(np.linalg.det(matrix), 1, rtol=rtol, atol=atol)
        )


def is_special_unitary(matrix: np.ndarray, *, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
    """Determines if a matrix is approximately unitary with unit determinant.

    A matrix is special-unitary if it is square and its adjoint is its inverse
    and its determinant is one.

    Args:
        matrix: The matrix to check.
        rtol: The per-matrix-entry relative tolerance on equality.
        atol: The per-matrix-entry absolute tolerance on equality.
    Returns:
        Whether the matrix is unitary with unit determinant within the given
        tolerance.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        return is_unitary(matrix, rtol=rtol, atol=atol) and (
            matrix.shape[0] == 0 or np.allclose(np.linalg.det(matrix), 1, rtol=rtol, atol=atol)
        )

def is_normal(matrix: np.ndarray, *, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
    """Determines if a matrix is approximately normal.

    A matrix is normal if it's square and commutes with its adjoint.

    Args:
        matrix: The matrix to check.
        rtol: The per-matrix-entry relative tolerance on equality.
        atol: The per-matrix-entry absolute tolerance on equality.

    Returns:
        Whether the matrix is normal within the given tolerance.
    """
    return matrix_commutes(matrix, matrix.T.conj(), rtol=rtol, atol=atol)


def matrix_commutes(
    m1: np.ndarray, m2: np.ndarray, *, rtol: float = 1e-5, atol: float = 1e-8
) -> bool:
    """Determines if two matrices approximately commute.

    Two matrices A and B commute if they are square and have the same size and
    AB = BA.

    Args:
        m1: One of the matrices.
        m2: The other matrix.
        rtol: The per-matrix-entry relative tolerance on equality.
        atol: The per-matrix-entry absolute tolerance on equality.

    Returns:
        Whether the two matrices have compatible sizes and a commutator equal
        to zero within tolerance.
    """
    return (
        m1.shape[0] == m1.shape[1]
        and m1.shape == m2.shape
        and np.allclose(m1.dot(m2), m2.dot(m1), rtol=rtol, atol=atol)
    )



