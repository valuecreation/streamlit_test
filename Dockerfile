# app/Dockerfile

FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y libgl1-mesa-glx && \
    apt-get install -y libglib2.0-0

RUN pip3 install -r requirements.txt

EXPOSE 8501

COPY . /app

ENTRYPOINT ["streamlit", "run", "--server.port=8501", "--server.address=0.0.0.0"]

CMD ["main.py"]