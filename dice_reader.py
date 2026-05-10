"""
dice_reader.py — Leitura do resultado de dados parados.

Lógica por tipo:
  D4   → face virada para BAIXO (normal mais próxima de Y-)
  D6   → face virada para CIMA  (normal mais próxima de Y+)
  D8   → face virada para CIMA  (normal mais próxima de Y+)
  D10  → vértice mais alto (maior Y no mundo) — o trapezoedro não tem face
         horizontal; cada vértice de pico identifica univocamente uma face.
  D12  → face virada para CIMA  (normal mais próxima de Y+)
  D20  → face virada para CIMA  (normal mais próxima de Y+)

⚠️  Os mapeamentos assumem que o OBJ está orientado com a face "1" apontando
    para +Y em repouso (convenção padrão de fabricantes de dados).
"""

import math
import numpy as np
import pybullet as pb


# ---------------------------------------------------------------------------
# Normais de face no espaço LOCAL
# ---------------------------------------------------------------------------

_s = 1.0 / math.sqrt(3)

FACE_NORMALS_D4 = np.array([
    [ _s,  _s,  _s],
    [ _s, -_s, -_s],
    [-_s,  _s, -_s],
    [-_s, -_s,  _s],
], dtype=np.float32)
FACE_VALUES_D4 = [1, 2, 3, 4]

FACE_NORMALS_D6 = np.array([
    [ 1,  0,  0],
    [-1,  0,  0],
    [ 0,  1,  0],
    [ 0, -1,  0],
    [ 0,  0,  1],
    [ 0,  0, -1],
], dtype=np.float32)
FACE_VALUES_D6 = [5, 3, 4, 2, 6, 1]

_o = 1.0 / math.sqrt(3)
FACE_NORMALS_D8 = np.array([
    [ _o,  _o,  _o], [ _o,  _o, -_o],
    [ _o, -_o,  _o], [ _o, -_o, -_o],
    [-_o,  _o,  _o], [-_o,  _o, -_o],
    [-_o, -_o,  _o], [-_o, -_o, -_o],
], dtype=np.float32)
FACE_VALUES_D8 = [1, 2, 3, 4, 5, 6, 7, 8]

# D10: mapeamento índice de vértice → valor da face
VERTEX_VALUES_D10 = {
    0: 1, 1: 3, 2: 5, 3: 7, 4: 9,   # vértices superiores
    5: 2, 6: 4, 7: 6, 8: 8, 9: 0,   # vértices inferiores (0 = 10)
}

_phi = (1 + math.sqrt(5)) / 2
_d12_raw = [
    [ 0,  1,  _phi], [ 0,  1, -_phi],
    [ 0, -1,  _phi], [ 0, -1, -_phi],
    [ 1,  _phi,  0], [ 1, -_phi,  0],
    [-1,  _phi,  0], [-1, -_phi,  0],
    [ _phi,  0,  1], [ _phi,  0, -1],
    [-_phi,  0,  1], [-_phi,  0, -1],
]
_d12_norm = math.sqrt(1 + _phi * _phi)
FACE_NORMALS_D12 = np.array(
    [[x / _d12_norm, y / _d12_norm, z / _d12_norm] for x, y, z in _d12_raw],
    dtype=np.float32
)
FACE_VALUES_D12 = list(range(1, 13))

_phi20 = (1 + math.sqrt(5)) / 2
_ico_verts_raw = []
for _s1 in (+1, -1):
    for _s2 in (+1, -1):
        _ico_verts_raw += [
            [0,              _s1,              _s2 * _phi20],
            [_s1,            _s2 * _phi20,     0           ],
            [_s1 * _phi20,   0,                _s2         ],
        ]
_ico_norm_val = math.sqrt(1 + _phi20 * _phi20)
_ico_verts = np.array(_ico_verts_raw, dtype=np.float64) / _ico_norm_val

_ICO_FACES = [
    (0,  4,  1), (0,  9,  4), (9,  5,  4), (4,  5,  8), (4,  8,  1),
    (8, 10,  1), (8,  3, 10), (5,  3,  8), (5,  2,  3), (2,  7,  3),
    (7, 10,  3), (7,  6, 10), (7, 11,  6), (11,  0,  6), (0,  1,  6),
    (6,  1, 10), (9,  0, 11), (9, 11,  2), (9,  2,  5), (7,  2, 11),
]
FACE_NORMALS_D20 = np.array(
    [(_ico_verts[a] + _ico_verts[b] + _ico_verts[c]) / 3.0
     for a, b, c in _ICO_FACES],
    dtype=np.float32
)
for _i in range(len(FACE_NORMALS_D20)):
    _l = np.linalg.norm(FACE_NORMALS_D20[_i])
    if _l > 1e-8:
        FACE_NORMALS_D20[_i] /= _l

FACE_VALUES_D20 = list(range(1, 21))


# ---------------------------------------------------------------------------
# Tabela de despacho
# ---------------------------------------------------------------------------

_FACE_DATA = {
    "d4":  (FACE_NORMALS_D4,  FACE_VALUES_D4),
    "d6":  (FACE_NORMALS_D6,  FACE_VALUES_D6),
    "d8":  (FACE_NORMALS_D8,  FACE_VALUES_D8),
    "d12": (FACE_NORMALS_D12, FACE_VALUES_D12),
    "d20": (FACE_NORMALS_D20, FACE_VALUES_D20),
}


# ---------------------------------------------------------------------------
# Funções internas
# ---------------------------------------------------------------------------

def _rotate_normal(local_normal: np.ndarray, quaternion) -> np.ndarray:
    """Aplica a rotação do quaternion Bullet a um vetor local → world-space."""
    rm = pb.getMatrixFromQuaternion(quaternion)
    R  = np.array(rm, dtype=np.float64).reshape(3, 3)
    return R @ local_normal


def _read_d10(orn) -> int:
    """D10: vértice mais alto no referencial do mundo."""
    from physics import _trapezoid_d10_verts
    local_verts = np.array(_trapezoid_d10_verts(r=1.0), dtype=np.float64)
    R = np.array(pb.getMatrixFromQuaternion(orn), dtype=np.float64).reshape(3, 3)
    world_verts = (R @ local_verts.T).T
    highest_idx = int(np.argmax(world_verts[:, 1]))
    return VERTEX_VALUES_D10.get(highest_idx, 0)


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

def read_die(dice_type: str, body_id: int, physics_client: int) -> int | None:
    """
    Lê o resultado de um dado parado.

    Parâmetros
    ----------
    dice_type     : "d4" | "d6" | "d8" | "d10" | "d12" | "d20"
    body_id       : ID do corpo rígido no PyBullet
    physics_client: ID do cliente PyBullet

    Retorna
    -------
    Valor inteiro da face ou None se o tipo for desconhecido.
    """
    _, orn = pb.getBasePositionAndOrientation(body_id,
                                              physicsClientId=physics_client)

    if dice_type == "d10":
        return _read_d10(orn)

    data = _FACE_DATA.get(dice_type)
    if data is None:
        print(f"[dice_reader] Tipo desconhecido: {dice_type!r}")
        return None

    face_normals, face_values = data
    target = np.array([0.0, -1.0, 0.0]) if dice_type == "d4" \
             else np.array([0.0,  1.0, 0.0])

    best_dot, best_value = -2.0, face_values[0]
    for local_n, value in zip(face_normals, face_values):
        dot = float(np.dot(_rotate_normal(local_n, orn), target))
        if dot > best_dot:
            best_dot, best_value = dot, value

    return best_value


def read_all_dice(dice_type: str,
                  dice_ids: list[int],
                  physics_client: int) -> list[int]:
    """
    Lê e imprime o resultado de todos os dados de uma rolagem.
    Retorna lista de valores na mesma ordem de dice_ids.
    """
    results = [read_die(dice_type, bid, physics_client) for bid in dice_ids]
    total   = sum(r for r in results if r is not None)
    labels  = " + ".join(str(r) for r in results)

    if len(results) == 1:
        print(f"[Resultado] {dice_type.upper()}: {results[0]}")
    else:
        print(f"[Resultado] {len(results)}× {dice_type.upper()}: {labels} = {total}")

    return results
