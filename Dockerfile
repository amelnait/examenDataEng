FROM ubuntu:22.04
COPY . /app
WORKDIR /app

RUN apt-get update && apt-get install -y python3 python3-pip
RUN pip3 install -r requirements.txt

CMD python main.py
