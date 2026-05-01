# AWS Lambda Python container image, x86_64.
# Lambda's runtime expects CMD ["module.handler"] format.
FROM public.ecr.aws/lambda/python:3.11

# Install PyTorch CPU only - cuts image size from ~3GB (with CUDA) to ~700MB.
RUN pip install --no-cache-dir \
        torch==2.2.0 \
        --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir numpy

# Copy model architecture from the model/ folder so the handler can load
# the trained checkpoint at startup.
COPY model/model.py model/tokenizer.py ${LAMBDA_TASK_ROOT}/

# Copy the Lambda handler.
COPY serving/handler.py ${LAMBDA_TASK_ROOT}/

# Model artifacts must be present at build time. The deploy script copies
# best.pt and tokenizer.json from out/ and data/ into ./model_artifacts/
# before docker build.
COPY model_artifacts/ ${LAMBDA_TASK_ROOT}/model_artifacts/

# Disable PyTorch threading parallelism (single-process Lambda env).
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV TOKENIZERS_PARALLELISM=false

# Telemetry / logging
ENV PYTHONUNBUFFERED=1

CMD ["handler.lambda_handler"]
