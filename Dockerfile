FROM python:3.9-slim

RUN echo 'Xin chao cac ban'

ARG ENVIRONMENT=server
ARG CLIENT_ID=1
ARG START_FILE=server.py

ENV ENVIRONMENT=${ENVIRONMENT}
ENV CLIENT_ID=${CLIENT_ID}
ENV LOG_LEVEL=INFO

WORKDIR /FED_SCP

COPY . /FED_SCP/

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 1883

CMD if [ "$ENVIRONMENT" = "server" ]; then \
        exec python ${START_FILE}; \
    else \
        exec python ${START_FILE} --id ${CLIENT_ID}; \
    fi
