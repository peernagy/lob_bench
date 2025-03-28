FROM python:3.10

WORKDIR /app

COPY requirements.txt /app

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

EXPOSE 8888

CMD ["/bin/bash"]