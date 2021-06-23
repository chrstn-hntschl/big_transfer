FROM tensorflow/tensorflow:2.2.0-gpu

LABEL maintainer="Christian Hentschel <christian.hentschel@hpi.de>"


ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get install -y \
  tzdata \
  vim \
  wget \
  unzip \
  ca-certificates \
  python3-pip \
  libyaml-cpp-dev && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists

ENV TZ=Europe/Berlin
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN pip3 --no-cache-dir install --upgrade pip 

# installing additional stuff
RUN pip3 install tensorflow-datasets==4.3.0 tensorflow-probability==0.10.0 PyYAML==5.4.1
RUN pip3 install cloudpickle==1.3.0

ADD https://github.com/chrstn-hntschl/big_transfer/archive/refs/heads/master.zip /tmp/big_transfer.zip
RUN cd /tmp && unzip big_transfer.zip && mv /tmp/big_transfer-master /opt/big_transfer

WORKDIR "/opt/big_transfer"

ENTRYPOINT ["python3", "-m", "bit_tf2.train"]
