## Agentes de Blackjack

### Contexto

El proyecto **blackjack-learner** entrena un agente basado en PyTorch para aprender estrategias básicas de Blackjack dentro de un entorno propio (`blackjack/environment.py`). El flujo principal se orquesta desde `main.py`, donde se exponen comandos de entrenamiento, visualización y juego interactivo.

### Resultados actuales

- Entrenamiento mediante un lazo simple de Q-learning (`blackjack/trainer.py`).
- Registro de métricas por episodio en `data/results.csv`.
- Visualización de recompensas con `visualization/graphs.py`.
- Automatizaciones para instalar dependencias, ejecutar pruebas y formatear código con `Makefile`.

### Próximos pasos sugeridos

- Incorporar múltiples mazos y reglas adicionales del juego.
- Implementar políticas más avanzadas (p. ej., modelos con memoria o redes recurrentes).
- Añadir tableros de control y visualizaciones interactivas.
- Registrar métricas adicionales (probabilidades estimadas, bankroll simulado).

### Principios de diseño

- **Modularidad:** cada responsabilidad vive en un módulo pequeño (`blackjack/`, `visualization/`, `docs/`).
- **Escalabilidad:** interfaces claras para extender agentes, utilidades o visualizaciones sin reescrituras masivas.
- **Consistencia:** mantener archivos acotados en tamaño para facilitar revisiones y seguir usando `uv` como gestor preferido.
- **Documentación en español:** README y documentación auxiliar deben conservar este idioma.
