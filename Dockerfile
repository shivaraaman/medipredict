# Works on Hugging Face Spaces (Docker SDK), Railway, Fly.io, or any Docker host.
FROM python:3.13-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=7860 DATABASE_PATH=/data/medipredict.db
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN mkdir -p /data && chmod 777 /data && useradd -m appuser
USER appuser
EXPOSE 7860
HEALTHCHECK CMD python -c "import urllib.request,os;urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/health')"
CMD gunicorn app:app --bind 0.0.0.0:${PORT} --workers 2 --timeout 60
