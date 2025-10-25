## Ejecución en Contenedores

### Construir imagen

```bash
make docker-build
```

Genera la imagen `blackjack-learner` definida en `Dockerfile`.

### Ejecutar entrenamiento

```bash
make docker-run EPISODES=500
```

Monta el código local y ejecuta `python main.py train`.

### Uso con Docker Compose

```bash
make compose-up
```

Inicia el servicio declarado en `docker-compose.yml`. Modifica variables como `EPISODES` mediante entorno:

```bash
EPISODES=1500 docker compose up --build
```

### Buenas prácticas

- Mantén el volumen montado para conservar resultados en `data/`.
- Actualiza dependencias con `uv lock` o `uv pip install` antes de reconstruir la imagen.
