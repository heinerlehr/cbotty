FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install uv for faster dependency management
RUN pip install uv

# Copy dependency files
COPY pyproject.toml .

# Copy source code
COPY src/ src/


# Expose port 8080 (NiceGUI default)
EXPOSE 8080

# Run the application
CMD ["bash", "-c", "uv run -m cbotty.cbotty"]