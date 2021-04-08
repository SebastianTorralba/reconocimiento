FROM python:3.6
RUN apt-get update
RUN python3 -m pip install --upgrade pip && rm -rf /var/cache/apk/*
ENV PYTHONUNBUFFERED 1
RUN apt-get install -y libblas-dev liblapack-dev  gfortran libleptonica-dev tesseract-ocr libtesseract-dev
RUN mkdir /app
WORKDIR /app
COPY ./app .
RUN cd /app
RUN pip3 install -r dependencias.txt
EXPOSE 8080

