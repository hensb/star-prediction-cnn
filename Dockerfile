FROM tensorflow/tensorflow:1.6.0-py3
MAINTAINER hendrik.schoeneberg@zuehlke.com

RUN mkdir /workspace
ADD resources/ /workspace
ADD predict_star_rating.py /workspace

WORKDIR /workspace

CMD ['python', '/workspace/predict_star_rating.py']