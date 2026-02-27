FROM python:3.12-slim

WORKDIR /app

# Install uv
RUN pip install --no-cache-dir uv

# Copy only the files needed to resolve dependencies
COPY pyproject.toml uv.lock* ./

# Install core (serving-only) dependencies — no torch/ML extras
RUN uv sync --frozen --no-install-project --no-extra generate --no-extra notebooks

# Copy application code
COPY app.py .
COPY templates/ templates/
COPY benchmarks/ benchmarks/

# forecasts/ is mounted at runtime via a volume
VOLUME ["/app/forecasts"]

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
