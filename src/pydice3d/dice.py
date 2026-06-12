"""
dice.py – Entidade de Domínio: Dado (PyBullet)

Responsabilidade: unir o corpo físico do PyBullet (body_id) com a malha
geométrica (DiceMesh), representando um dado completo.

Posição e orientação são sempre consultadas via bullet.getBasePositionAndOrientation,
nunca armazenadas localmente — PyBullet é a fonte da verdade para estado físico.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional

import pybullet as pb

from pydice3d.dice_mesh import DiceMesh, DiceType, get_mesh


DEFAULT_SCALE: float = 1.0


@dataclass
class Dice:
    """
    Dado poliédrico: body_id PyBullet + geometria.

    Atributos
    ---------
    body_id   : identificador do corpo no mundo PyBullet
    mesh      : malha geométrica imutável (vértices, faces, normais)
    dice_type : string identificadora ("d6", "d20", etc.)
    scale     : escala visual aplicada
    """
    body_id:   int
    mesh:      DiceMesh
    dice_type: str
    scale:     float = DEFAULT_SCALE

    @classmethod
    def create(
        cls,
        dice_type: DiceType,
        position:  tuple | list | np.ndarray,
        physics,                               # PhysicsWorld
        scale:     float = DEFAULT_SCALE,
        name:      str   = "",
    ) -> "Dice":
        """
        Cria um dado registrando seu corpo no PhysicsWorld fornecido.

        Parâmetros
        ----------
        dice_type : "d4", "d6", "d8", "d10", "d12" ou "d20"
        position  : posição inicial no mundo (x, y, z)
        physics   : instância de PhysicsWorld
        scale     : fator de escala visual
        """
        body_id = physics.create_dice_body(dice_type, position, scale)
        mesh    = get_mesh(dice_type, scale=scale)
        return cls(body_id=body_id, mesh=mesh, dice_type=dice_type, scale=scale)

    # ------------------------------------------------------------------
    # Estado físico (sempre via PyBullet)
    # ------------------------------------------------------------------

    @property
    def position(self) -> np.ndarray:
        pos, _ = pb.getBasePositionAndOrientation(self.body_id)
        return np.array(pos, dtype=np.float32)

    @property
    def orientation_quat(self) -> np.ndarray:
        """Quaternion [x, y, z, w] — formato nativo do PyBullet."""
        _, orn = pb.getBasePositionAndOrientation(self.body_id)
        return np.array(orn, dtype=np.float64)

    @property
    def orientation_matrix(self) -> np.ndarray:
        """Matriz de rotação 3×3 derivada do quaternion do PyBullet."""
        xyzw = self.orientation_quat
        # Converte [x,y,z,w] → [w,x,y,z] para quat_to_matrix
        w, x, y, z = xyzw[3], xyzw[0], xyzw[1], xyzw[2]
        return _quat_wxyz_to_matrix(w, x, y, z)
        

    # ------------------------------------------------------------------
    # Conveniências
    # ------------------------------------------------------------------

    @property
    def num_faces(self) -> int:
        return self.mesh.num_faces

    @property
    def num_vertices(self) -> int:
        return self.mesh.num_vertices

    def __repr__(self) -> str:
        pos = self.position
        return (f"Dice({self.dice_type}, "
                f"pos=({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}), "
                f"scale={self.scale})")


# ---------------------------------------------------------------------------
# Helper interno — conversão quaternion → matriz
# ---------------------------------------------------------------------------
# dice.py recebe o quaternion do PyBullet como (pos, orn) onde orn=[x,y,z,w].
# orientation_matrix desempacota para w,x,y,z antes de chamar quat_to_matrix.

from pydice3d.math_utils import quat_to_matrix as _quat_to_matrix


def _quat_wxyz_to_matrix(w: float, x: float, y: float, z: float) -> np.ndarray:
    """Shim de compatibilidade: aceita componentes separados, delega a math_utils."""
    return _quat_to_matrix(np.array([x, y, z, w], dtype=np.float64))