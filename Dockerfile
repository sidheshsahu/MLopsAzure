# FROM python:3.10-slim

# ENV PYTHONDONTWRITEBYTECODE=1
# ENV PYTHONUNBUFFERED=1

# WORKDIR /app

# COPY requirements.txt .

# RUN pip install --upgrade pip && pip install -r requirements.txt

# COPY . .

# EXPOSE 80

# CMD ["python", "app.py"]



FROM python:3.10-slim


ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

EXPOSE 80

CMD ["uvicorn", "app:app","--host", "0.0.0.0", "--port","80"]


