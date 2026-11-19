FROM python:3.13-slim

# uv installs the exact package versions from uv.lock
COPY --from=ghcr.io/astral-sh/uv:0.11.26 /uv /usr/local/bin/uv

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1 \
    MLFLOW_DISABLE_AGENT_HINT=1 \
    PATH="/app/.venv/bin:$PATH"

# Install dependencies first: this layer is cached until pyproject/uv.lock change
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY src ./src
COPY app ./app
COPY .streamlit ./.streamlit

EXPOSE 8000
# 0.0.0.0 so the API is reachable from outside the container
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
