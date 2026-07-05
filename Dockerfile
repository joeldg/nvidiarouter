# @spec[PROJECT_PROFILE.md#Requirements]
# Container image for NVIDIA-SmartRoute-CLI gateway.
FROM python:3.11-slim

# Avoid interactive prompts and .pyc clutter; unbuffered logs.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install the package first (leverages layer caching for deps).
COPY pyproject.toml README.md ./
COPY nvidia_smartroute ./nvidia_smartroute
RUN pip install --upgrade pip && pip install .

# The gateway listens on 9000 (0.0.0.0:9000).
EXPOSE 9000

# Provide credentials at runtime, e.g.:
#   docker run -p 9000:9000 -e NVIDIA_API_KEY=nvapi-... nvidia-smartroute
CMD ["nvidia-smartroute", "start", "--host", "0.0.0.0", "--port", "9000"]
