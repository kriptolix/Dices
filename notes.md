## Dependencies
pillow
PyOpenGL
PyOpenGL-accelerate
numpy
scipy 
pybullet 
pygobject
simpleaudio

## Problemas 

* adicionar capacidades de carregamento de texturas
* audio com simpleaudio

## Decisões

em physics._make_collision_shape, r = DICE_TARGET_SIZE, o valor antigo era a metade mas dava muitos problemas de colisão. Futuramente cada dado deve ter um valor separado, assim o ajuste mais fino é possível

## Correções

### Melhorias

1. Rim Light (alto retorno)

Adicione um brilho sutil nas bordas voltadas para longe da câmera.

float rim = 1.0 - max(dot(N, V), 0.0);
rim = pow(rim, 3.0);

color += rim * vec3(0.3, 0.3, 0.4);

Isso dá volume e destaca a silhueta do dado.

2. Hemispheric Lighting

Em vez de um ambient constante:

float hemi = N.y * 0.5 + 0.5;

vec3 sky    = vec3(0.5, 0.6, 0.8);
vec3 ground = vec3(0.15, 0.12, 0.1);

vec3 ambient = mix(ground, sky, hemi);

As sombras ficam muito mais agradáveis.

3. Toon/Soft Diffuse

Substituir o Lambert puro:

float diff = max(dot(N,L),0.0);
diff = smoothstep(0.0, 1.0, diff);

ou

diff = diff * diff * (3.0 - 2.0 * diff);

Reduz a sensação de iluminação "seca".

4. Fresnel

Muito usado em materiais estilizados.

float fresnel = pow(1.0 - max(dot(N,V),0.0), 5.0);

color += fresnel * vec3(0.15);

Dá uma sensação de material mais rico.

5. Oren-Nayar (substitui Lambert)

É um modelo difuso mais suave para superfícies rugosas.

Visualmente costuma parecer melhor que Lambert mesmo sem PBR.

6. Gradiente procedural no dado

Sem textura:

float h = v_frag_pos.y;
vec3 base = mix(colorA, colorB, h);

ou baseado na normal:

vec3 base = mix(darkColor, lightColor, N.y*0.5+0.5);

### Testes a fazer

1. Reduzir a intensidade especular

+ spec * u_light_color * 0.15;

2. Aplicar um limite suave

spec = pow(max(dot(N, H), 0.0), u_shininess);
spec = smoothstep(0.2, 1.0, spec);

3. Aumentar a luz ambiente
Se hoje ela é algo como:

u_ambient = (0.1, 0.1, 0.1)

experimente:

u_ambient = (0.25, 0.25, 0.25)

Isso reduz o contraste excessivo das sombras.

4. Gamma correction
Se você ainda não faz:

color = pow(color, vec3(1.0/2.2));

antes do frag_color.


## Construção de detecção de colisões para tocar sons.

O motor de física detecta:

início da colisão;
intensidade do impacto;
tipo de superfície.

Então ele envia um evento para o sistema de áudio.

Fluxo conceitual
Física → Evento de colisão → Sistema de áudio → Som
O que detectar

Para cada frame:

1. Detectar colisão nova

Somente quando um contato começa.

Evite tocar som:

enquanto objetos continuam encostados;
em micro vibrações.

2. Calcular intensidade do impacto

Use:

velocidade relativa;
impulso da colisão;
energia transferida.

Exemplo:

impact = relative_velocity.length()

ou idealmente:

impact = collision_impulse

3. Aplicar threshold

Ignorar colisões pequenas:

if impact < 0.3:
    return

4. Converter impacto em áudio

Mapeie impacto para:

volume;
pitch;
escolha de sample.

Exemplo:

volume = clamp(impact / 10.0, 0.1, 1.0)

pitch = random(0.95, 1.05)

Pequena variação de pitch evita repetição artificial.

Separar tipos de som
Rolling

Som contínuo:

enquanto o dado desliza/gira;
volume depende da velocidade angular.
Collision

Som curto:

trigger instantâneo;
baseado no impacto.

. Pitch randomizado (essencial)
pitch = random(0.92, 1.08)


2. Volume baseado em impacto
volume = clamp(impact / 10.0, 0.1, 1.0)
3. EQ simples (muito importante)
impactos leves → mais médios/agudos
impactos fortes → mais graves
4. Layering (muito eficaz)

Em vez de muitos arquivos:

“hit base” (corpo do som)
“clack” (transiente curto)
“rumble” (grave opcional)

Combinados dinamicamente.

5. Micro delays (desalinhamento humano)
delay = random(0.0, 0.02)

Evita som “robotizado”.


