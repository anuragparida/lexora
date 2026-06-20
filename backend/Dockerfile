FROM python:3.12-slim

WORKDIR /app

RUN pip install uv

COPY pyproject.toml .
RUN uv sync

COPY . .

ENV DATABASE_URL=sqlite:///./data/vocabeo_words.db

EXPOSE 8000

CMD ["uv", "run", "python", "main.py"]
