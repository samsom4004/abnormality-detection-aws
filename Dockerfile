# CPU serving container (SageMaker BYOC contract: /ping and /invocations on 8080).
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MODEL_DIR=/opt/ml/model
WORKDIR /app
RUN pip install --no-cache-dir torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[serve]" && useradd --create-home appuser
USER appuser
EXPOSE 8080
CMD ["uvicorn", "abnormality_detection.serve:app", "--host", "0.0.0.0", "--port", "8080"]
