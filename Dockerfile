FROM python:3.12-slim AS base
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1
COPY pyproject.toml README.md ./
COPY gateway ./gateway
RUN pip install --no-cache-dir . \
 && mkdir -p /app/data \
 && useradd -r -u 10001 mmaas && chown -R mmaas /app
USER mmaas
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz',timeout=2).status==200 else 1)"
ENTRYPOINT ["mmaas", "--config", "/app/gateway.yaml"]
CMD ["serve"]
