FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

# Many platforms set $PORT; default to 7860 (common on Spaces) if not set.
EXPOSE 7860

CMD ["sh", "-c", "gunicorn -b 0.0.0.0:${PORT:-7860} --timeout 120 app:app"]


