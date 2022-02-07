FROM apache/tika:latest-full

ENV DEBIAN_FRONTEND=noninteractive

COPY requirements.txt .

RUN apt-get update -yq && apt-get -yq install --no-install-recommends tesseract-ocr-rus python3-pip \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get -yq purge python3-pip

RUN apt-get -yq autoremove && apt-get -yq clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* requirements.txt
