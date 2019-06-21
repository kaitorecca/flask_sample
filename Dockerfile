FROM ubuntu:latest
MAINTAINER Tai "hongtai91@gmail.com"
RUN apt-get update -y
RUN apt-get install -y python3-pip python3-dev build-essential
RUN apt update && apt install -y libsm6 libxext6
RUN apt-get -y install tesseract-ocr
COPY . /app
WORKDIR /app
RUN pip install pillow
RUN pip install pytesseract
RUN pip install opencv-contrib-python
RUN pip install -r requirements.txt
ENTRYPOINT ["python"]
CMD ["service.py"]