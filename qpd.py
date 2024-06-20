import numpy as np
from numpy import sin, cos
from conditions import *
from kak import *

from typing import Union, TYPE_CHECKING
if TYPE_CHECKING:
    from numpy.typing import DTypeLike, ArrayLike

from typing import Tuple, Callable, List, Optional

import cirq

class KakLikeDecomp():
    """Description of two-qubit unitary for cutting
    Any two qubit operation can be stored in form 
    U = g · (a1 ⊗ a0) · (u0 · I + u1 · X + u2 · Y + u3 · Z) · (b1 ⊗ b0)
    where sum_i(abs(u_i)^2) = 1

    """
    def __init__(
        self,
        *,
        global_phase: complex = complex(1),
        single_qubit_operations_before: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        amplitudes: Tuple[complex, complex, complex, complex],
        single_qubit_operations_after: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> None:
        """Initializes a decomposition for a two-qubit operation U.

        U = g · (a1 ⊗ a0) · (u0·II + u1·XX + u2·YY + u3·ZZ)) · (b1 ⊗ b0)

        Args:
            global_phase: g from the above equation.
            single_qubit_operations_before: b0, b1 from the above equation.
            amplitudes: u0, u1, u2, u3 from the above equation.
            single_qubit_operations_after: a0, a1 from the above equation.
        """
        self.global_phase: complex = global_phase
        self.single_qubit_operations_before: Tuple[np.ndarray, np.ndarray] = (
            single_qubit_operations_before
            or (np.eye(2, dtype=np.complex64), np.eye(2, dtype=np.complex64))
        )
        self.amplitudes = amplitudes
        self.single_qubit_operations_after: Tuple[np.ndarray, np.ndarray] = (
            single_qubit_operations_after
            or (np.eye(2, dtype=np.complex64), np.eye(2, dtype=np.complex64))
        )
    
    def gamma_factor(self):
        """Computes the quantity sum_{k != k'} |u_k||u_k'|.

        Args:
            u: A 1D numpy array.

        Returns:
            A float representing the quantity.
        """
        u = self.amplitudes
        # Compute the absolute value of each element in u.
        abs_u = np.abs(u)

        # Compute the outer product of abs_u with itself.
        outer_product = np.outer(abs_u, abs_u)

        # Set the diagonal of the outer product to 0.
        outer_product[np.diag_indices_from(outer_product)] = 0

        # Sum the outer product.
        sum_outer_product = np.sum(outer_product)

        return 1 + 2 * sum_outer_product

def kak_like_decomposition(unitary_object:
                    Union[np.ndarray, KakDecomposition, KakLikeDecomp],
                    rtol: float = 1e-5,
                    atol: float = 1e-8,
                    check_preconditions: bool = True,):
    if isinstance(unitary_object, KakDecomposition):
        theta = unitary_object.interaction_coefficients
        u_0 = cos(theta[0]) * cos(theta[1]) * cos(theta[2]) + 1j * sin(theta[0]) * sin(theta[1]) * sin(theta[2])
        u_1 = 1j* sin(theta[0]) * cos(theta[1]) * cos(theta[2]) + cos(theta[0]) * sin(theta[1]) * sin(theta[2])
        u_2 = 1j* cos(theta[0]) * sin(theta[1]) * cos(theta[2]) + sin(theta[0]) * cos(theta[1]) * sin(theta[2])
        u_3 = sin(theta[0]) * sin(theta[1]) * cos(theta[2]) + 1j * cos(theta[0]) * cos(theta[1]) * sin(theta[2])
        u = np.array([u_0, u_1, u_2, u_3])
        b = unitary_object.single_qubit_operations_before
        a = unitary_object.single_qubit_operations_after
        g = unitary_object.global_phase
        qpd = KakLikeDecomp(global_phase=g,
                                     single_qubit_operations_before=b,
                                     amplitudes=u,
                                     single_qubit_operations_after=a)
        return qpd
    if isinstance(unitary_object, np.ndarray):
        kak = kak_decomposition(unitary_object, rtol=rtol, atol=atol, check_preconditions= check_preconditions)
        theta = kak.interaction_coefficients
        u_0 = cos(theta[0]) * cos(theta[1]) * cos(theta[2]) + 1j * sin(theta[0]) * sin(theta[1]) * sin(theta[2])
        u_1 = 1j* sin(theta[0]) * cos(theta[1]) * cos(theta[2]) + cos(theta[0]) * sin(theta[1]) * sin(theta[2])
        u_2 = 1j* cos(theta[0]) * sin(theta[1]) * cos(theta[2]) + sin(theta[0]) * cos(theta[1]) * sin(theta[2])
        u_3 = sin(theta[0]) * sin(theta[1]) * cos(theta[2]) + 1j * cos(theta[0]) * cos(theta[1]) * sin(theta[2])
        u = np.array([u_0, u_1, u_2, u_3])
        b = kak.single_qubit_operations_before
        a = kak.single_qubit_operations_after
        g = kak.global_phase
        qpd = KakLikeDecomp(global_phase=g,
                                     single_qubit_operations_before=b,
                                     amplitudes=u,
                                     single_qubit_operations_after=a)
        return qpd
    if isinstance(unitary_object, KakLikeDecomp):
        return unitary_object
    

def simple_subcircuit(q0, q1, 
                      local_i, 
                      a_0, a_1,
                      b_0, b_1):
    """
    Having KAK-like decomposition of two qubit gate:
    U = a_0 \otimes a_1 sum_i u_i local_i \otimes local_i b_0 \otimes b_1 
    constructs simple (without ancillas) subcircuit for the corresponding subcircuit in QPD,
    specified by local gate in interaction term of KAK-like decomposition

    Args:
        q0, q1: qubits that initial two qubit gate acts on
        local_i: local gate from KAK-like decomposition
        a_0, a_1: local gates which act after interaction term of KAK
        b_0, b_1: local gates which act before interaction term of KAK

    Returns:
        circ_gate: subcircuit that will replace given two qubit gate when cimulating cut circuit
    """
    circ_gate = cirq.Circuit()
    circ_gate.append(cirq.Moment([b_0(q0),b_1(q1)]))
    circ_gate.append(cirq.Moment([local_i(q0),local_i(q1)]))
    circ_gate.append(cirq.Moment([a_0(q0),a_1(q1)]))

    return circ_gate

def ancilla_subcirc(q0, q1, 
                    anc_0, anc_1, 
                    b_0, b_1,
                    a_0, a_1,
                    phi, local_i, local_j):
    """
    Having KAK-like decomposition of two qubit gate:
    U = a_0 \otimes a_1 sum_i u_i local_i \otimes local_i b_0 \otimes b_1 
    constructs subcircuit with ancillas for the corresponding circuit in QPD,
    specified by local gate in interaction term of KAK-like decomposition

    Args:
        q0, q1: qubits that initial two qubit gate acts on
        anc_0, anc_1: ancillary qubits used to obtain sign of the corresponding weight
        local_i: local gate from KAK-like decomposition
        a_0, a_1: local gates which act after interaction term of KAK
        b_0, b_1: local gates which act before interaction term of KAK
        phi: relative phase to apply on each ancilla
        local_i, local_j: local gates from KAK-like decomposition

    Returns:
        circ_gate: subcircuit that will replace given two qubit gate when cimulating cut circuit
    """
    circ_gate = cirq.Circuit()
    circ_gate.append([cirq.reset(anc_0),cirq.reset(anc_1)])
    circ_gate.append(cirq.H.on_each([anc_0, anc_1]))

    phase = np.array([[1, 0],
                    [0, np.exp(-1j*phi/2)]])
    p_gate = cirq.MatrixGate(phase, name = 'phi='+str(np.angle(phase[-1,-1]).round(2)))
    circ_gate.append(p_gate.on_each([anc_0, anc_1]))

    circ_gate.append(b_0(q0))
    circ_gate.append(b_1(q1))

    circ_gate.append([cirq.X(anc_0), cirq.X(anc_1)])
    circ_gate.append(cirq.ControlledGate(local_i)(anc_0, q0))
    circ_gate.append(cirq.ControlledGate(local_i)(anc_1, q1))
    circ_gate.append([cirq.X(anc_0), cirq.X(anc_1)])

    circ_gate.append(cirq.ControlledGate(local_j)(anc_0, q0))
    circ_gate.append(cirq.ControlledGate(local_j)(anc_1, q1))

    circ_gate.append(a_0(q0))
    circ_gate.append(a_1(q1))

    circ_gate.append(cirq.H.on_each([anc_0, anc_1]))

    #circ_gate.append(cirq.measure_each([anc_0, anc_1]))
    #circ_gate.append(cirq.measure([anc_0, anc_1]))
    return circ_gate
