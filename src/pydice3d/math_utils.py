"""
math_utils.py – Utilitários Matemáticos

Funções de álgebra linear e quaternions usadas em toda a lib.
Nenhum outro módulo de pydice3d deve reimplementar estas funções.

Convenção de quaternions
────────────────────────
Todos os quaternions seguem o formato PyBullet: [x, y, z, w] (ndarray).
"""

from __future__ import annotations

import math
import numpy as np


# ────────────────────────────────────────────────────────────────────────────
# Vetores
# ────────────────────────────────────────────────────────────────────────────

def normalize(v: np.ndarray) -> np.ndarray:
    """Normaliza vetor N-D. Retorna vetor zero se norma < epsilon."""
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else np.zeros_like(v)


# ────────────────────────────────────────────────────────────────────────────
# Quaternions  (formato PyBullet: [x, y, z, w])
# ────────────────────────────────────────────────────────────────────────────

def quat_to_matrix(xyzw: np.ndarray) -> np.ndarray:
    """
    Quaternion [x, y, z, w] → matriz de rotação 3×3 (float64).

    Formato de entrada: PyBullet / SciPy (escalar w no índice 3).
    """
    x, y, z, w = xyzw
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


def quat_slerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """
    Interpolação esférica (slerp) entre dois quaternions [x, y, z, w].

    Parâmetros
    ----------
    a, b : quaternions de partida e chegada
    t    : fator de interpolação [0, 1]  (0 → a, 1 → b)

    Retorna quaternion normalizado.
    """
    dot = float(np.dot(a, b))
    if dot < 0.0:           # escolhe o caminho mais curto
        b   = -b
        dot = -dot
    dot = min(1.0, dot)

    if dot > 0.9995:        # quaternions quase paralelos — lerp linear
        result = a + t * (b - a)
        return result / np.linalg.norm(result)

    theta_0 = math.acos(dot)
    theta   = theta_0 * t
    sin_t   = math.sin(theta)
    sin_t0  = math.sin(theta_0)
    s0 = math.cos(theta) - dot * sin_t / sin_t0
    s1 = sin_t / sin_t0
    result = s0 * a + s1 * b
    return result / np.linalg.norm(result)