FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install uv for faster dependency management
RUN pip install uv

# Copy dependency files
COPY pyproject.toml .

# Copy source code and environment
COPY src/ src/
COPY config/ config/
COPY .env .env

# Expose port 8080 (NiceGUI default)
EXPOSE 8080

ENV PYTHONPATH=/app/src

# Install package and run build-time tasks
RUN uv pip install --system -e . && \
    uv run -m synthetic.sgen && \
    uv run -m chromatools.initdb task=init

# Run the application
CMD ["bash", "-c", "uv run -m cbotty.cbotty"]