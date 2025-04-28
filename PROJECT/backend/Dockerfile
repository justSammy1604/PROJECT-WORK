# Backend Dockerfile
# Stage 1: Builder - Install dependencies and copy code
FROM python:3.10-slim-bookworm AS builder
WORKDIR /app

# Copy only requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
# Using --prefix=/install ensures packages are installed in a predictable location
# that we can easily copy later.
RUN pip install --no-cache-dir --prefix="/install" -r requirements.txt

# Now copy the rest of the application code
COPY . .

# Stage 2: Final runtime image - Copy installed packages and application code
FROM python:3.10-slim-bookworm
WORKDIR /app

# Copy installed packages from the specific location in the builder stage
# Adjust the python version path (3.10) if your base image changes significantly
COPY --from=builder /install /usr/local/

# Copy application code from the builder stage
COPY --from=builder /app /app

EXPOSE 4200

# Set the command to run the application
CMD ["python", "server.py"]