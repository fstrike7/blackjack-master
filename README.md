## Blackjack Learner

Proyecto en Python que entrena un agente basado en PyTorch para aprender a jugar Blackjack mediante refuerzo sencillo. El entrenamiento registra métricas en CSV y dispone de herramientas de visualización con matplotlib.

### Estructura del proyecto

```
blackjack-learner/
├── blackjack/
│   ├── __init__.py
│   ├── agent.py
│   ├── environment.py
│   ├── trainer.py
│   └── utils.py
├── data/
│   ├── checkpoints/
│   └── results.csv
├── visualization/
│   └── graphs.py
├── main.py
├── requirements.txt
└── README.md
```

### Instalación

1. Crear y activar un entorno virtual (opcional pero recomendado).
2. Instalar dependencias (preferentemente con `uv`):

```bash
uv pip sync pyproject.toml
```

   Alternativa con `pip`:

```bash
pip install -r requirements.txt
```

### Uso del CLI

- Entrenar al agente:

```bash
python main.py train --episodes 1000
```

- Visualizar los resultados (abre una ventana con el gráfico):

```bash
python main.py visualize
```

  Para guardar la figura sin mostrarla:

```bash
python main.py visualize --save plots/rewards.png --no-show
```

- Jugar manualmente una ronda:

```bash
python main.py play
```

### Makefile y tooling

- `make install` instala dependencias con uv por defecto (`PKG_MANAGER=pip` para usar pip).
- `make install-dev` agrega dependencias de desarrollo (pytest, ruff, sympy).
- `make test`, `make format` y `make lint` ejecutan pruebas básicas y formateo/lint con Ruff.
- `make train EPISODES=2000`, `make visualize` y `make play` ejecutan los subcomandos del CLI.
- `make docker-run` construye la imagen y la ejecuta, o usa `docker compose up --build` para un flujo equivalente.

### Documentación adicional

- `Agents.md` resume contexto, resultados y próximos pasos del agente.
- La carpeta `docs/` contiene guías adicionales (`tooling.md`, `docker.md`) para nuevas implementaciones.

### Notas

- Los resultados se guardan en `data/results.csv`.
- Los checkpoints del modelo se almacenan en `data/checkpoints/`.
- Ajusta hiperparámetros como `--epsilon-decay` y `--learning-rate` desde la CLI de entrenamiento.
