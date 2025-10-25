## Guía de Tooling

### Dependencias con `uv`

1. Instalar dependencias básicas:
   ```bash
   uv pip sync
   ```
2. Instalar dependencias de desarrollo (pytest, ruff, sympy):
   ```bash
   uv pip sync
   uv pip install -r requirements-dev.txt
   ```
3. En caso de preferir `pip`, usar `make install PKG_MANAGER=pip`.

### Formateo y lint

- `make format` ejecuta `ruff format .`.
- `make lint` ejecuta `ruff check .`.
- Ajusta configuraciones adicionales en `pyproject.toml` si necesitas reglas específicas.

### Pruebas

- Ejecuta `make test` para validar importaciones básicas y cálculos de soporte.
- Las pruebas utilizan `pytest` y `sympy`.

### Recomendaciones

- Mantén archivos por debajo de ~300 líneas para facilitar revisiones.
- Evita comentarios triviales; prioriza código y docstrings concisos.
