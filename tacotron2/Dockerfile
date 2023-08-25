FROM redhat/ubi8:latest
WORKDIR app/
RUN yum install python3.11 python3.11-pip libsndfile -y
RUN pip3 install mecab-python3
RUN pip3 install unidic-lite
ENV NUMBA_CACHE_DIR /data
COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r ./requirements.txt
COPY download.py ./
RUN python3 download.py
COPY . ./
CMD ["uvicorn", "predict:app", "--host", "0.0.0.0"]