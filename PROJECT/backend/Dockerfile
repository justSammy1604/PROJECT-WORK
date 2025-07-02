# Backend Dockerfile
# Stage 1: Builder - Install dependencies and copy code
FROM python:3.10-slim-bookworm AS builder
WORKDIR /app

# Install CA certificates (needed for SSL/TLS connections, e.g., Groq API)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir --prefix="/install" -r requirements.txt

# Now copy the rest of the application code
COPY . .

# Stage 2: Final runtime image - Copy installed packages and application code
FROM python:3.10-slim-bookworm
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl \
    libnss3 libatk-bridge2.0-0 libxcomposite1 libxrandr2 libgbm1 \
    libgtk-3-0 libxdamage1 libxfixes3 libxshmfence1 libasound2 libx11-xcb1 \
    fonts-liberation libappindicator3-1 xdg-utils && \
    update-ca-certificates && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local/
COPY --from=builder /app /app

RUN playwright install --with-deps

ENV REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

EXPOSE 4200
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "4200"]