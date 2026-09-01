FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    XDG_CACHE_HOME=/home/surgivision/.cache

WORKDIR /app

RUN apt-get update \
    && apt-get install --no-install-recommends -y libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /usr/sbin/nologin surgivision

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY app ./app
COPY scripts ./scripts

RUN python -m pip install . \
    && mkdir -p /home/surgivision/.cache \
    && chown -R surgivision:surgivision /home/surgivision /app

USER surgivision

EXPOSE 8501

CMD ["streamlit", "run", "app/dashboard.py", "--server.address=0.0.0.0", "--server.port=8501"]
