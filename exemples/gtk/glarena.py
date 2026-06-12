"""
glarena.py - GTK4 GLArea: integração entre GTK e DiceSimulation

Responsabilidade: conectar os sinais do ciclo de vida GTK/GL
(realize / unrealize / render / resize) ao DiceSimulation e ao Renderer.

glarena conhece apenas:
  - pydice3d.simulation  (DiceSimulation, RollResult)
  - pydice3d.renderer    (Renderer — necessário na camada GL)
  - debug_wire           (wireframes de colisão, local ao frontend)

Modos de debug
──────────────
  DEBUG_NONE      : renderização normal
  DEBUG_COLLISION : só wireframe do hull de colisão (sem mesh visual)
  DEBUG_OVERLAY   : mesh visual + wireframe de colisão sobrepostos
"""

from __future__ import annotations

import json
import numpy as np
from importlib.resources import files

import gi
gi.require_version("Gtk", "4.0")
from gi.repository import Gtk

from OpenGL import GL

from pydice3d.simulation import DiceSimulation, RollResult
from pydice3d.renderer   import Renderer

from debug_wire import (
    CollisionWireframe, build_wire_program,
    DEBUG_NONE, DEBUG_COLLISION, DEBUG_OVERLAY,
)


# ────────────────────────────────────────────────────────────────────────────
# Caminhos dos assets (pertencem à camada GTK: só aqui sabe onde estão)
# ────────────────────────────────────────────────────────────────────────────

_ATLAS_DIR  = files("pydice3d.assets").joinpath("atlas")
_ATLAS_NPY  = str(_ATLAS_DIR.joinpath("atlas.npy"))
_ATLAS_JSON = _ATLAS_DIR.joinpath("atlas.json")


# ────────────────────────────────────────────────────────────────────────────
# DiceGLArea
# ────────────────────────────────────────────────────────────────────────────

class DiceGLArea(Gtk.GLArea):
    """
    Widget GTK4 que exibe a simulação de dados em um contexto OpenGL 3.3.

    API pública
    ───────────
    start_simulation(spec)  inicia nova rolagem; spec = {"d6": 2, "d20": 1}
    simulation              acesso direto ao DiceSimulation (leitura)
    simulating              True enquanto os dados estão em movimento
    debug_mode              DEBUG_NONE | DEBUG_COLLISION | DEBUG_OVERLAY
    on_roll_complete        callback(RollResult) chamado quando todos param
    """

    def __init__(self) -> None:
        super().__init__()

        self.set_required_version(3, 3)
        self.set_has_depth_buffer(True)
        self.set_focusable(True)

        # Dimensões do framebuffer em pixels físicos.
        # NÃO usar get_allocated_width/height em _on_render: em HiDPI eles
        # retornam pixels lógicos e o viewport cobriria só 1/4 da área.
        self._vp_w: int = 660
        self._vp_h: int = 460

        self._sim = DiceSimulation(on_result=self._on_roll_complete)

        # Recursos OpenGL — criados em realize, destruídos em unrealize
        self._renderer:  Renderer | None = None
        self._wire_prog: int = 0
        self._wire_objs: list[CollisionWireframe] = []
        self._atlas_json: dict | None = None

        self._debug_mode: int = DEBUG_NONE

        # Callback externo opcional: AppWindow pode sobrescrever
        self.on_roll_complete: object = None   # callable(RollResult) | None

        self.theme = "dark"

        # Conecta sinais GTK
        self.connect("realize",   self._on_realize)
        self.connect("unrealize", self._on_unrealize)
        self.connect("render",    self._on_render)
        self.connect("resize",    self._on_resize)

    # ── Propriedades públicas ─────────────────────────────────────────────────

    @property
    def simulation(self) -> DiceSimulation:
        return self._sim

    @property
    def simulating(self) -> bool:
        return self._sim.is_rolling

    @property
    def debug_mode(self) -> int:
        return self._debug_mode

    @debug_mode.setter
    def debug_mode(self, mode: int) -> None:
        self._debug_mode = mode
        if self._renderer:
            self._renderer.debug_mode = mode
        self.queue_render()

    @property
    def theme(self) -> str:
        return self._sim.theme

    @theme.setter
    def theme(self, value: str) -> None:
        self._sim.theme = value          # simulation reconstrói a cena internamente
        if self._renderer:
            self._renderer.theme = value
        self.queue_render()

    # ── Sinais GTK ────────────────────────────────────────────────────────────

    def _on_resize(self, _area, width: int, height: int) -> None:
        self._vp_w = max(width, 1)
        self._vp_h = max(height, 1)
        self._sim.resize(self._vp_w, self._vp_h)

    def _on_realize(self, _area) -> None:
        self.make_current()
        if self.get_error():
            return

        self._wire_prog = build_wire_program()

        # Carrega atlas de glifos — I/O feito aqui onde o contexto GL existe.
        # O Renderer só é criado em start_simulation(), quando a cena já existe.
        try:
            with open(_ATLAS_JSON, "r", encoding="utf-8") as f:
                self._atlas_json = json.load(f)
        except Exception as e:
            print(f"[AVISO] Não foi possível carregar atlas.json: {e}")

    def _on_unrealize(self, _area) -> None:
        self.make_current()
        if self._renderer:
            self._renderer.delete()
            self._renderer = None
        for w in self._wire_objs:
            w.delete()
        self._wire_objs.clear()
        if self._wire_prog:
            GL.glDeleteProgram(self._wire_prog)
            self._wire_prog = 0

    def _on_render(self, _area, _ctx) -> bool:
        w, h = self._vp_w, self._vp_h

        # Avança física — scene.update() é chamado internamente em step()
        self._sim.step()

        scene = self._sim.scene
        if self._renderer and scene:
            VP      = self._sim.view_projection()
            cam_pos = self._sim.camera_position()

            if self._debug_mode == DEBUG_COLLISION:
                GL.glViewport(0, 0, w, h)
                GL.glClearColor(0.0, 0.0, 0.0, 0.0)
                GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
                self._draw_wire(VP, scene)

            elif self._debug_mode == DEBUG_OVERLAY:
                self._renderer.draw(scene, VP, cam_pos, w, h)
                GL.glEnable(GL.GL_POLYGON_OFFSET_LINE)
                GL.glPolygonOffset(-1.0, -1.0)
                self._draw_wire(VP, scene)
                GL.glDisable(GL.GL_POLYGON_OFFSET_LINE)

            else:
                self._renderer.draw(scene, VP, cam_pos, w, h)
        else:
            GL.glViewport(0, 0, w, h)
            GL.glClearColor(0.0, 0.0, 0.0, 0.0)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        return True

    # ── API pública ───────────────────────────────────────────────────────────

    def start_simulation(self, spec: dict[str, int]) -> None:
        """
        Inicia uma nova rolagem, descartando a anterior.

        Parâmetros
        ----------
        spec : dicionário {tipo: quantidade}, ex: {"d6": 2, "d100": 1}.
               Tipos suportados: d4, d6, d8, d10, d12, d20, d100, df.
               d100 adiciona automaticamente 1 d10 parceiro.
        """
        self.make_current()
        if self.get_error():
            return

        # Libera wireframes do roll anterior
        for w in self._wire_objs:
            w.delete()
        self._wire_objs.clear()

        # Inicia simulação — spawn, estados, monitor e RenderScene
        self._sim.roll(spec, theme=self._sim.theme)

        # Cria um CollisionWireframe por dado usando dice_types da simulação
        for dtype in self._sim.dice_types:
            self._wire_objs.append(CollisionWireframe(dtype))

        # Recarrega recursos GPU do renderer com a cena recém-criada
        if self._renderer is None:
            self._renderer = Renderer(
                self._sim.scene, self._sim.dice_types,
                atlas_npy=_ATLAS_NPY,
                atlas_json=self._atlas_json,
                theme=self._sim.theme,
            )
        else:
            self._renderer.reload(self._sim.scene, self._sim.dice_types)

        if self._renderer:
            self._renderer.debug_mode = self._debug_mode

        self.grab_focus()

    # ── Internos ──────────────────────────────────────────────────────────────

    def _draw_wire(self, VP: np.ndarray, scene) -> None:
        """Desenha os hulls de colisão de todos os dados ativos."""
        if not self._wire_prog:
            return
        for wire, rd in zip(self._wire_objs, scene.dice_renders):
            MVP = (VP @ rd.model_mat).astype(np.float32)
            wire.draw(MVP, self._wire_prog)

    def _on_roll_complete(self, result: RollResult) -> None:
        """Recebe o resultado do DiceSimulation e repassa ao callback externo."""
        print(f"[RESULT] {result.summary()}")
        if callable(self.on_roll_complete):
            self.on_roll_complete(result)