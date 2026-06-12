# pydice3d — Architecture Reference

This document describes the internal design of the library. Its purpose is to let module-level and class-level docstrings stay short, covering only what the code itself cannot express.

---

## Module responsibilities

| Module | Responsibility |
|---|---|
| `simulation.py` | Orchestrator. The only entry point a frontend needs. |
| `physics.py` | PyBullet world, rigid bodies, tray walls, collision polling. |
| `spawner.py` | Spawn and launch a set of dice: positions, velocities, torques. |
| `dice.py` | Dice entity: ties a PyBullet body ID to a `DiceMesh`. |
| `dice_state.py` | Per-die lifecycle state machine. Reads from PyBullet, writes nothing back. |
| `dice_mesh.py` | Immutable polyhedron geometry: vertices, faces, normals, face values. No physics, no rendering. |
| `results.py` | Aggregate roll results from a list of `DiceState`. Detect roll completion. |
| `scene.py` | CPU-side render data: `DiceRenderData`, `RenderScene`, UV generation, glyph mapping, themes. |
| `renderer.py` | OpenGL 3.3 renderer: VAO/VBO/EBO management, MSDF glyph atlas, draw calls. |
| `shaders.py` | GLSL source strings, shader compilation, uniform helpers. |
| `camera.py` | Orbital camera expressed in spherical coordinates. Produces view and projection matrices. |
| `math_utils.py` | Quaternion and vector math. Single source of truth for the whole codebase. |
| `audio.py` | Collision and rolling audio engine. Driven by physics events from `simulation`. |

---

## Dependency graph

Arrows mean "imports from". No cycles.

```
math_utils          (no internal deps)
dice_mesh           → math_utils
dice                → dice_mesh, math_utils
dice_state          → dice, math_utils
results             → dice_state, math_utils
scene               → dice_state, shaders             # DiceTheme lives here
renderer            → scene, shaders
camera              (no internal deps)
audio               (no internal deps)
shaders             (no internal deps)
physics             (no internal deps)
spawner             → dice, dice_state
simulation          → physics, camera, dice_state,
                      spawner, results, scene, audio
```

`glarena` (GTK demo, not part of the lib):
```
glarena             → simulation, renderer            # only two lib imports
```

---

## Lifecycle of a roll

### 1. Spawn — `simulation.roll(spec, theme)`

`spawner.spawn_dice` creates one `Dice` per entry in `spec`, registers each with `PhysicsWorld`, computes clustered initial positions, and applies random velocities and torques via `resetBaseVelocity`. Returns a `SpawnResult` with the list of `DiceState` objects.

`simulation` also calls `RenderScene.from_states` at this point, building CPU-side vertex buffers and UV data for every die. GPU objects (`DiceGpuObject`) are created when `renderer.reload` is called by the frontend.

### 2. Step — `simulation.step()`

Called once per render frame. Internally runs `steps_per_tick` PyBullet substeps (default: 4). After each substep:

- `DiceState.update_status()` — reads linear and angular velocities from PyBullet, drives the state machine.
- Collision events are collected for audio.

After all substeps:

- `RenderScene.update(states, alpha)` — writes model matrices from current orientations.
- `RollMonitor.tick()` — checks if all dice are `RESTING`; fires `on_result` once if so.
- Audio engine receives collision events and rolling state.

### 3. Result — `RollResult.from_states(states)`

Called by `RollMonitor` when all dice are resting. Reads the face value of each die by finding the face whose world-space normal is most aligned with `+Y` (or `−Y` for d4). d100 combines the tens die with its d10 partner: `result = tens + units`, with `00 + 0 = 100`.

---

## Dice lifecycle state machine

```
SPAWNED ──► ROLLING ──► SETTLING ──► RESTING
               ▲            │
               └────────────┘  (jitter resets to ROLLING)
```

Transitions are driven entirely by velocity thresholds read from PyBullet:

| Threshold | Value |
|---|---|
| Linear speed (XZ plane) | 0.05 m/s |
| Angular speed | 0.10 rad/s |
| Frames below threshold to enter SETTLING | 20 |
| Consecutive frames below threshold to reach RESTING | 30 |
| Total settling frames timeout (stacked dice) | 180 |

---

## Face value reading

- **All dice except d4**: face with normal most aligned with world `+Y`.
- **d4**: face with normal most aligned with world `−Y` (result is read from the bottom face, as on a physical d4).
- **df (Fudge)**: same as d6 (top face). Opposite faces carry the same value: `+1/+1`, `−1/−1`, `0/0`.
- **d100**: two-die system. `tens ∈ {0, 10, …, 90}` from the d100 die, `units ∈ {0, …, 9}` from the paired d10 (face value 10 is treated as 0). Combined: `tens + units`; if both are 0, result is 100.

All face value reading logic lives in `results.py` (`read_face_value`). `dice.py` and `dice_state.py` do not compute results.

---

## Render pipeline

```
DiceMesh (geometry, face values)
    │
    ▼
DiceRenderData.from_state(state, theme)
    │  Triangulates faces, computes per-vertex UVs,
    │  maps face values to glyph indices.
    ▼
RenderScene
    │  Holds list of DiceRenderData.
    │  update(states, alpha) writes model matrices.
    ▼
Renderer.draw(scene, VP, cam_pos, w, h)
    │  Uploads uniforms, binds MSDF atlas,
    │  issues one draw call per die.
```

`DICE_VISUAL_SCALE` in `renderer.py` scales each die type visually without affecting the PyBullet collision shape.

### MSDF glyph atlas

The atlas is a pre-generated `.npy` file (RGBA uint8) with a matching `.json` descriptor. `shaders.py` parses the JSON and builds a `float32 (N, 4)` UV table (`u0, v0, u1, v1` per glyph). The fragment shader samples the atlas using the face's glyph index and applies multi-channel SDF rendering.

Glyph index convention:

| Range | Content |
|---|---|
| 0–9 | Digits 0–9 |
| 10–20 | Two-digit numbers 10–20 |
| 21–30 | d100 tens: 00, 10, 20, …, 90 |
| 31 | `+` (Fudge) |
| 32 | `−` (Fudge) |
| 33 | blank face (Fudge) |
| 255 | no glyph |

---

## Camera

`Camera` stores position as spherical coordinates `(azimuth_deg, elevation_deg, radius)` around a `target` point. This avoids gimbal lock at normal viewing angles and makes orbit/zoom/pan trivial to implement.

`view_matrix()` always uses `up = [0, 1, 0]`. Elevation is clamped to `[2°, 88°]` to prevent the singularity that occurs when the camera is directly above the target (`forward ∥ up`).

---

## Quaternion convention

All quaternions throughout the codebase use the PyBullet format: `[x, y, z, w]` (scalar component last). The single canonical implementation is `math_utils.quat_to_matrix(xyzw)`.

---

## Themes

`DiceTheme` and `DICE_THEMES` are defined in `scene.py` (presentation layer). `renderer.py` imports from there. `dice_mesh.py` has no knowledge of visual presentation.

Available themes: `"light"` (white dice, dark glyphs), `"dark"` (dark dice, light glyphs).

Changing the theme at runtime via `simulation.theme = "dark"` or `renderer.theme = "dark"` updates colors immediately without rebuilding GPU buffers.

---

## d4 special rendering

The d4 displays three numbers per face (one per edge), following the convention of physical d4s where you read the number at the bottom edge. Implementation:

Each triangular face is split into three sub-triangles by the centroid. Sub-triangle `k` covers the edge opposite vertex `k` and displays the value of the face opposite to global vertex `k`. The expanded glyph array has `12` slots (`4 faces × 3 sub-triangles`).

---

## SpawnConfig

Controls the spawn behavior of `spawn_dice`. Key parameters:

| Parameter | Default | Effect |
|---|---|---|
| `spawn_height` | 1.8 | Y position at birth |
| `spawn_cluster_xz` | (0, 3.5) | Center of the spawn cluster |
| `cluster_radius` | 4.0 | Max radius from center |
| `min_separation` | 2.2 | Minimum distance between die centers |
| `throw_azimuth_deg` | 270° | Direction of throw (−Z = into tray) |
| `throw_spread_deg` | 30° | Fan spread around azimuth |
| `speed_min/max` | 3.5–4.5 m/s | Horizontal throw speed range |
| `vy_max` | 1.0 m/s | Max upward velocity component |
| `torque_max` | 7.0 rad/s | Max initial angular velocity |
| `seed` | None | Fixed seed for reproducibility |

---

## What docstrings should still cover

After this document exists, docstrings in the codebase should be limited to:

- **Public API methods with non-obvious parameters** — a one-line summary and parameter list if needed.
- **Non-trivial algorithms** — a brief note on why, not what (the code shows what).
- **Gotchas and constraints** — things that would surprise a reader (e.g. "elevation clamped to avoid singularity").

Module-level docstrings can be reduced to a single line stating the module's responsibility. Everything else lives here.
