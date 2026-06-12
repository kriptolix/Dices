"""
dice_state.py – Estado e Ciclo de Vida do Dado (PyBullet)

Responsabilidade: manter o ciclo de vida de cada dado (SPAWNED → ROLLING →
SETTLING → RESTING) e expor orientação/velocidade via PyBullet.

Leitura de valores de face é responsabilidade de results.py, que consome
DiceState como dado de entrada e calcula resultados de rolagem.

Ciclo de vida: SPAWNED → ROLLING → SETTLING → RESTING
"""

from __future__ import annotations

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pybullet as pb
import math

from pydice3d.dice import Dice


from pydice3d.math_utils import quat_to_matrix, quat_slerp


# ────────────────────────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────────────────────────


SETTLING_LINEAR_THRESHOLD:  float = 0.05   # m/s
SETTLING_ANGULAR_THRESHOLD: float = 0.10   # rad/s

SETTLING_FRAMES_REQUIRED: int = 20
RESTING_FRAMES_REQUIRED:  int = 30

SETTLING_TIMEOUT_FRAMES: int = 180   # ~3 s a 60 Hz


class DiceStatus(Enum):
    SPAWNED  = auto()
    ROLLING  = auto()
    SETTLING = auto()
    RESTING  = auto()


# ────────────────────────────────────────────────────────────────────────────
# DiceState
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class DiceState:
    """
    Estado mutável de ciclo de vida de um dado no mundo PyBullet.

    Orientação e velocidades são sempre lidas do PyBullet — não há estado
    angular local. O único estado mantido aqui são os contadores de
    estabilização e o resultado final.

    Para interpolação de frames pelo renderer, prev_orientation é salvo
    a cada update_status (quaternion [x,y,z,w] do PyBullet).
    """
    dice: Dice

    status: DiceStatus = DiceStatus.SPAWNED

    # Quaternions [x,y,z,w] para interpolação de render
    prev_orientation: np.ndarray = field(
        default_factory=lambda: np.array([0., 0., 0., 1.])
    )

    _settling_frames: int = field(default=0, repr=False)
    _resting_frames:  int = field(default=0, repr=False)
    # Contador acumulado de frames em SETTLING (não zera ao voltar para ROLLING).
    # Usado pelo timeout para forçar RESTING em dados empilhados com jitter persistente.
    _settling_total:  int = field(default=0, repr=False)

    # ── construtor semântico ─────────────────────────────────────────

    @classmethod
    def create(cls, dice: Dice) -> "DiceState":
        """Cria estado inicial para um dado já registrado no PyBullet."""
        _, orn = pb.getBasePositionAndOrientation(dice.body_id)
        return cls(
            dice=dice,
            prev_orientation=np.array(orn, dtype=np.float64),
        )

    # ── leitura de estado do PyBullet ────────────────────────────────

    @property
    def orientation_quat(self) -> np.ndarray:
        """Quaternion atual [x,y,z,w] lido do PyBullet."""
        _, orn = pb.getBasePositionAndOrientation(self.dice.body_id)
        return np.array(orn, dtype=np.float64)

    @property
    def rotation_matrix(self) -> np.ndarray:
        """Matriz de rotação 3×3 atual."""
        return quat_to_matrix(self.orientation_quat)

    @property
    def linear_velocity(self) -> np.ndarray:
        lin, _ = pb.getBaseVelocity(self.dice.body_id)
        return np.array(lin, dtype=np.float64)

    @property
    def angular_velocity(self) -> np.ndarray:
        _, ang = pb.getBaseVelocity(self.dice.body_id)
        return np.array(ang, dtype=np.float64)

    # ── estabilização ────────────────────────────────────────────────

    def update_status(self) -> None:
        """
        Updates lifecycle by reading speeds from PyBullet. Also
        saves prev_orientation for renderer interpolation.        
        """
        if self.status == DiceStatus.RESTING:
            return

        # Salva orientação anterior para interpolação
        self.prev_orientation = self.orientation_quat

        lin = self.linear_velocity
        ang = self.angular_velocity

        lin_speed_xz = math.sqrt(float(lin[0])**2 + float(lin[2])**2)
        ang_speed    = float(np.linalg.norm(ang))
        moving       = (lin_speed_xz > SETTLING_LINEAR_THRESHOLD or
                        ang_speed     > SETTLING_ANGULAR_THRESHOLD)

        if self.status == DiceStatus.SPAWNED:
            self.status = DiceStatus.ROLLING
            return

        if moving:
            self.status           = DiceStatus.ROLLING
            self._settling_frames = 0
            self._resting_frames  = 0
            # _settling_total não é zerado: conta tempo total com baixa velocidade,
            # mesmo que intercalado com breves picos de jitter.
            return

        # Dado abaixo dos limiares de movimento — acumula os dois contadores.
        self._settling_frames += 1
        self._settling_total  += 1

        if self.status == DiceStatus.ROLLING:
            if self._settling_frames >= SETTLING_FRAMES_REQUIRED:
                self.status          = DiceStatus.SETTLING
                self._resting_frames = 0
            return

        if self.status == DiceStatus.SETTLING:
            self._resting_frames += 1
            # Caminho normal: frames quietos consecutivos suficientes.
            if self._resting_frames >= RESTING_FRAMES_REQUIRED:
                self.status = DiceStatus.RESTING
                return
            # Caminho de timeout: dado com jitter persistente (tipicamente
            # empilhado). Aceita o resultado mesmo sem frames consecutivos
            # suficientes.
            if self._settling_total >= SETTLING_TIMEOUT_FRAMES:
                self.status = DiceStatus.RESTING

    # ── conveniências ────────────────────────────────────────────────

    @property
    def is_resting(self) -> bool:
        return self.status == DiceStatus.RESTING

    @property
    def world_vertices(self) -> np.ndarray:
        return self.dice.world_vertices(self.rotation_matrix)

    @property
    def world_face_normals(self) -> np.ndarray:
        return self.dice.world_face_normals(self.rotation_matrix)

    def interpolated_rotation_matrix(self, alpha: float) -> np.ndarray:
        """
        Interpola entre prev_orientation e a orientação atual via slerp.
        alpha=0 → frame anterior, alpha=1 → frame atual.
        Usado pelo renderer para suavizar entre steps de física.
        """
        q = quat_slerp(self.prev_orientation, self.orientation_quat, alpha)
        return quat_to_matrix(q)

    def __repr__(self) -> str:
        return (f"DiceState({self.dice.dice_type}, "
                f"status={self.status.name})")