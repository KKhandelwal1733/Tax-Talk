FROM python:3.11-slim

# 1. Install uv directly from the official image (fastest method)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# 2. Set strict environment variables
# This explicitly tells uv where to build the environment and adds it to the Linux PATH
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_PROJECT_ENVIRONMENT="/app/.venv" \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# 3. Create the non-root user early
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser

# 4. Copy ONLY the dependency files first for perfect caching
COPY pyproject.toml uv.lock README.md ./

# 5. Install dependencies (but NOT the project yet)
# This layer caches perfectly. It will only rebuild if you add a new package.
RUN uv sync --frozen --no-dev --no-install-project

# 6. Copy your source code
COPY src ./src

# 7. Install the actual tax-talk project
RUN uv sync --frozen --no-dev

# 8. Grant the non-root user ownership of the pre-built environment and code
RUN chown -R appuser:appgroup /app

# 9. Switch to the secured non-root user
USER appuser

EXPOSE 8000

# 10. The pure runtime command. 
# No 'uv', no cache checks. Because we modified the PATH in step 2, 
# Linux knows exactly where uvicorn is and runs it instantly.
CMD ["uvicorn", "tax_talk.api.main:app", "--host", "0.0.0.0", "--port", "8000"]