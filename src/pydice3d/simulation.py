"""
simulation.py – Data Simulation Orchestrator

Responsibilities
─────────────────
- Create and destroy the PhysicsWorld
- Execute spawn_dice and maintain the DiceState list
- Advance the simulation frame by frame (step)
- Manage the RenderScene lifecycle for OpenGL frontends
- Calculate camera/projection matrices
- Resize the physical tray when the viewport changes
- Monitor the end of the roll via RollMonitor
- Display the final result via `result`

Headless use (tests, CLI, server)
──────────────────────────────────────
    from pydice3d.simulation import DiceSimulation

    sim = DiceSimulation()
    sim.roll({"d6": 2, "d20": 1})
    while not sim.is_done:
        sim.step()
    print(sim.result.as_dict())   # {"d6": [3, 5], "d20": [17]}

OpenGL frontend use
────────────────────────
    from pydice3d.simulation import DiceSimulation, RollResult

    sim = DiceSimulation(on_result=my_callback)
    sim.resize(viewport_w, viewport_h)
    sim.roll({"d6": 3}, theme="dark")   # cria/recria RenderScene internamente

    # a cada frame do loop de render:
    sim.step()                           # fisica + atualiza poses da cena
    scene    = sim.scene                 # RenderScene com model_mats atualizados
    VP       = sim.view_projection()     # mat4 float32
    cam_pos  = sim.camera_position()     # vec3 float32
    renderer.draw(scene, VP, cam_pos, w, h)

    # tipos dos dados para recursos GPU (wireframe, VAOs, etc.):
    dice_types = sim.dice_types          # ["d6", "d6", "d20"]
"""

from __future__ import annotations

import math
from typing import Optional, Callable, TYPE_CHECKING

import numpy as np

from pydice3d.physics    import PhysicsWorld
from pydice3d.dice_state import DiceState
from pydice3d.spawner    import spawn_dice, SpawnConfig
from pydice3d.results    import RollMonitor, RollResult
from pydice3d.camera     import Camera
from pydice3d.audio      import DiceAudioEngine
from pydice3d.scene      import RenderScene

# Reexportado para que frontends não precisem importar de results diretamente
__all__ = ["DiceSimulation", "RollResult"]

if TYPE_CHECKING:
    from pydice3d.audio import CollisionEvent
    
# Quantos substeps de física são executados por chamada a step()
STEPS_PER_TICK: int = 4


# ────────────────────────────────────────────────────────────────────────────
# DiceSimulation
# ────────────────────────────────────────────────────────────────────────────

class DiceSimulation:
    """
    Orquestrador da simulação de dados.

    Parâmetros
    ----------
    on_result : callable(RollResult) opcional.
                Chamado exatamente uma vez quando todos os dados param.
                Se None, use a propriedade `result` após `is_done == True`.
    steps_per_tick : quantos steps de física por chamada a step().
                     Padrão: 4 — bom equilíbrio entre velocidade e suavidade.
    spawn_cfg : SpawnConfig personalizado. Usa padrão se None.
    """

    def __init__(
        self,
        on_result:      Optional[Callable[[RollResult], None]] = None,
        steps_per_tick: int = STEPS_PER_TICK,
        spawn_cfg:      Optional[SpawnConfig] = None,
    ) -> None:
        self._physics       = PhysicsWorld()
        self._states:       list[DiceState]      = []
        self._monitor:      Optional[RollMonitor] = None
        self._on_result     = on_result
        self._steps_per_tick = steps_per_tick
        self._spawn_cfg     = spawn_cfg
        
        self.audio = DiceAudioEngine()

        # Câmera — delegada inteiramente ao módulo camera.py.
        # A câmera original usava eye=[0,12,0] com up=[0,0,-1], uma vista
        # top-down pura. Camera.from_eye_target com esse eye produz
        # elevation=90°, que causa singularidade no look_at (forward ∥ up).
        # Usamos coordenadas esféricas diretamente: elevation=89° preserva
        # a sensação top-down sem degeneração, com azimuth=90° (câmera em +Z).
        self._camera = Camera(
            target        = np.array([0.0, 0.0, 0.0], dtype=float),
            azimuth_deg   = 90.0,
            elevation_deg = 89.0,
            radius        = 12.0,
            fov_y_deg     = 35.0,
            near          = 0.1,
            far           = 50.0,
        )

        # Viewport (pixels)
        self._vp_w: int = 660
        self._vp_h: int = 460

        self._simulating: bool = False

        # Cena de render — criada em roll(), atualizada em step().
        # None em modo headless (sem chamada a roll(theme=...)).
        self._scene:  RenderScene | None = None
        self._theme:  str = "light" 

    # ── Configuration ─────────────────────────────────────────────────────────

    def resize(self, width: int, height: int) -> None:
        """
        Notifica a simulação do tamanho do viewport.

        Recalcula os limites da bandeja física para que ela preencha o
        frustum da câmera sem deixar espaço vazio ou cortar os dados.
        Deve ser chamado sempre que o widget de render muda de tamanho,
        antes ou depois de roll().
        """
        self._vp_w = max(width, 1)
        self._vp_h = max(height, 1)

        # Half-height da bandeja no plano Y=0 projetada pelo frustum
        eye_height = float(self._camera.eye_position()[1])
        half_h = math.tan(math.radians(self._camera.fov_y_deg / 2)) * eye_height
        aspect = self._vp_w / self._vp_h
        half_w = half_h * aspect

        # 5 % de margem para que as paredes fiquem fora do frustum
        self._physics.resize_tray(half_w * 0.95, half_h * 0.95)

    def set_camera(
        self,
        eye:     Optional[np.ndarray] = None,
        target:  Optional[np.ndarray] = None,
        fov_deg: Optional[float] = None,
        near:    Optional[float] = None,
        far:     Optional[float] = None,
    ) -> None:
        """
        Ajusta parâmetros da câmera sem exigir todos.

        `eye` e `target` recalculam azimute/elevação/raio internamente via
        Camera.from_eye_target — a câmera continua expressada em coordenadas
        esféricas, preservando orbit/zoom/pan.
        """
        if eye is not None or target is not None:
            new_eye    = np.asarray(eye,    dtype=float) if eye    is not None else self._camera.eye_position()
            new_target = np.asarray(target, dtype=float) if target is not None else np.asarray(self._camera.target, dtype=float)
            self._camera = Camera.from_eye_target(
                eye       = new_eye,
                target    = new_target,
                fov_y_deg = fov_deg if fov_deg is not None else self._camera.fov_y_deg,
                near      = near    if near    is not None else self._camera.near,
                far       = far     if far     is not None else self._camera.far,
            )
            # from_eye_target pode produzir elevation=90° se eye está
            # diretamente acima do target — clamp para evitar singularidade.
            
            self._camera.elevation_deg = float(
                np.clip(self._camera.elevation_deg,
                         self._camera._ELEV_MIN,
                         self._camera._ELEV_MAX)
            )
        else:
            if fov_deg is not None: self._camera.fov_y_deg = float(fov_deg)
            if near    is not None: self._camera.near      = float(near)
            if far     is not None: self._camera.far       = float(far)

    # ── Controle da rolagem ──────────────────────────────────────────────────

    def roll(
        self,
        spec:      dict[str, int],
        cfg:       Optional[SpawnConfig] = None,
        on_result: Optional[Callable[[RollResult], None]] = None,
        theme:     Optional[str] = None,
    ) -> None:
        """
        Inicia uma nova rolagem, descartando qualquer rolagem anterior.

        Parâmetros
        ----------
        spec      : dicionário {tipo: quantidade}, ex: {"d6": 2, "d20": 1}.
                    d100 adiciona automaticamente 1 d10 de unidades por dado.
        cfg       : SpawnConfig para esta rolagem (sobrescreve o padrão do __init__).
        on_result : callback para esta rolagem específica (sobrescreve o do __init__).
        theme     : tema visual ("light" | "dark"). Mantém o anterior se None.
                    Quando fornecido, (re)cria a RenderScene automaticamente.
                    Em modo headless, pode ser omitido — scene permanece None.
        """
        # Limpa estado anterior
        self._simulating = False
        self._physics.remove_all_dice()
        self._states.clear()
        self._monitor = None
        self._scene   = None

        if theme is not None:
            self._theme = theme

        effective_cfg = cfg or self._spawn_cfg or SpawnConfig()
        result        = spawn_dice(spec=spec, physics=self._physics, cfg=effective_cfg)
        self._states  = result.states

        # Cria RenderScene se houver tema definido (modo GL) ou se já existia uma
        # antes (reroll sem alterar tema explicitamente).
        if self._theme or theme is not None:
            self._scene = RenderScene.from_states(self._states, self._theme)

        callback = on_result or self._on_result
        self._monitor = RollMonitor(self._states, on_complete=callback)
        self._simulating = True

    def step(self) -> None:
        """
        Avança a simulação por um tick (steps_per_tick substeps de física).

        Chame uma vez por frame do loop de render. Quando todos os dados
        param, `is_done` passa a True e o callback on_result é disparado.
        """
        if not self._simulating or not self._states:
            return

        # Coleta eventos de colisão dentro de cada substep — não depois.
        # Com SIM_SUBSTEPS=6 um impacto pode durar apenas 1-2 substeps;
        # consultar só no final perderia a maioria dos contatos.
        collision_events = []
        for _ in range(self._steps_per_tick):
            pre_vel = self._physics._snapshot_velocities()
            self._physics.step()
            for s in self._states:
                s.update_status()
            collision_events.extend(
                self._physics.poll_collision_events(pre_vel)
            )

        # Áudio: dispara apenas o evento de maior impulso por par
        # (evita múltiplos disparos do mesmo impacto em substeps consecutivos)
        best: dict[tuple[int, int], CollisionEvent] = {}
        for evt in collision_events:
            pair = (min(evt.body_a, evt.body_b), max(evt.body_a, evt.body_b))
            if pair not in best or evt.impulse > best[pair].impulse:
                best[pair] = evt
        for evt in best.values():
            self.audio.on_collision(evt)

        # Áudio: loop contínuo de rolling
        self.audio.on_rolling(self._states)
        self.audio.tick()

        # Atualiza poses da cena de render com as orientações atuais (alpha=1:
        # sem interpolação — a interpolação é opcional e pode ser feita pelo
        # frontend chamando scene.update(states, alpha) diretamente se precisar).
        if self._scene is not None:
            self._scene.update(self._states, alpha=1.0)

        if self._monitor:
            self._monitor.tick()

        if self._physics.all_sleeping():
            self._simulating = False
            self.audio.on_roll_complete()

    def stop(self) -> None:
        """Interrompe a simulação sem limpar os dados (preserva poses finais)."""
        self._simulating = False
        self.audio.stop_all()

    def reset(self) -> None:
        """Remove todos os dados e reseta o estado completo."""
        self._simulating = False
        self._physics.remove_all_dice()
        self._states.clear()
        self._monitor = None
        self._scene   = None
        self.audio.stop_all()

    # ── Estado e resultado ───────────────────────────────────────────────────

    @property
    def is_rolling(self) -> bool:
        """True enquanto a simulação está ativa (dados ainda em movimento)."""
        return self._simulating

    @property
    def is_done(self) -> bool:
        """True quando todos os dados pararam e o resultado está disponível."""
        return self._monitor is not None and self._monitor.completed

    @property
    def result(self) -> Optional[RollResult]:
        """
        Resultado final da rolagem. None se ainda não concluída.
        Acesse após is_done == True, ou use on_result para ser notificado.
        """
        return self._monitor.result if self._monitor else None

    @property
    def partial_result(self) -> Optional[RollResult]:
        """Resultado parcial com os dados que já pararam. Útil para HUD progressivo."""
        return self._monitor.partial_result() if self._monitor else None

    @property
    def progress(self) -> float:
        """Fração de dados parados [0.0, 1.0]."""
        return self._monitor.progress if self._monitor else 0.0

    @property
    def states(self) -> list[DiceState]:
        """Lista de DiceState dos dados ativos (leitura)."""
        return self._states

    @property
    def scene(self) -> "RenderScene | None":
        """
        Cena de render com model_mats atualizados a cada step().

        None em modo headless (roll() chamado sem theme).
        Passe ao Renderer diretamente — sem precisar importar RenderScene.
        """
        return self._scene

    @property
    def dice_types(self) -> list[str]:
        """
        Lista de tipos dos dados ativos, na mesma ordem que scene.dice_renders.

        Ex: ["d6", "d6", "d20"]
        Útil para alocar recursos GPU (VAOs, wireframes) sem navegar em states.
        """
        return [s.dice.dice_type for s in self._states]

    @property
    def theme(self) -> str:
        """Tema visual atual ("light" | "dark")."""
        return self._theme

    @theme.setter
    def theme(self, value: str) -> None:
        """
        Altera o tema visual em tempo de execução.

        Se já existe uma cena, reconstrói a RenderScene com o novo tema
        (cores de glyphs, etc.) sem precisar reiniciar a rolagem.
        """
        self._theme = value
        if self._scene is not None and self._states:
            self._scene = RenderScene.from_states(self._states, self._theme)

    @property
    def physics(self) -> PhysicsWorld:
        """Acesso direto ao PhysicsWorld (para usos avançados)."""
        return self._physics

    # ── Áudio ─────────────────────────────────────────────────────────────────

    @property
    def audio_enabled(self) -> bool:
        """True se o motor de áudio está ativo."""
        return self.audio.enabled

    @audio_enabled.setter
    def audio_enabled(self, value: bool) -> None:
        """
        Liga ou desliga o áudio em tempo de execução.

        Quando desligado, todos os sons em andamento são parados imediatamente
        e nenhum novo som é gerado enquanto a simulação avança.

        Exemplo::

            sim.audio_enabled = False   # mudo
            sim.audio_enabled = True    # religa
        """
        self.audio.enabled = bool(value)
        if not self.audio.enabled:
            self.audio.stop_all()

    @property
    def audio_volume(self) -> float:
        """Volume global do áudio [0.0, 1.0]."""
        return self.audio.master_volume

    @audio_volume.setter
    def audio_volume(self, value: float) -> None:
        """
        Ajusta o volume global do áudio sem desabilitar o motor.

        O valor é aplicado imediatamente a todas as colisões e ao loop de
        rolling dos próximos ticks. Sons já em reprodução não são afetados
        (eles usam o volume fixado no momento do disparo).

        Exemplo::

            sim.audio_volume = 0.5    # metade do volume
            sim.audio_volume = 0.0    # silencioso sem desligar o motor
        """
        
        self.audio.master_volume = float(np.clip(value, 0.0, 1.0))

    # ── Camera ───────────────────────────────────────────────────────────────

    @property
    def camera(self) -> Camera:
        """Acesso direto à instância Camera (orbit, zoom, pan)."""
        return self._camera

    def view_matrix(self) -> np.ndarray:
        """Matriz view 4×4 float32 (look-at)."""
        return self._camera.view_matrix()

    def projection_matrix(self) -> np.ndarray:
        """Matriz de projeção perspectiva 4×4 float32."""
        return self._camera.projection_matrix(self._vp_w, self._vp_h)

    def view_projection(self) -> np.ndarray:
        """Produto P×V como float32 — pronto para enviar ao shader."""
        return self._camera.view_projection(self._vp_w, self._vp_h)

    def camera_position(self) -> np.ndarray:
        """Posição da câmera no espaço do mundo (vec3 float32)."""
        return self._camera.position

    # ── Ciclo de vida do PhysicsWorld ────────────────────────────────────────

    def __del__(self) -> None:
        # PhysicsWorld já faz pb.disconnect no próprio __del__,
        # mas garantimos reset limpo se o objeto for coletado cedo.
        try:
            self.reset()
        except Exception:
            pass