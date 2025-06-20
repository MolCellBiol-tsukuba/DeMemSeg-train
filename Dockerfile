# ベースイメージの指定
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Dockerfile内で設定する環境変数
ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    TZ=Asia/Tokyo \
    DEBIAN_FRONTEND=noninteractive \
    PYENV_ROOT=/opt/pyenv \
    PATH="/opt/pyenv/bin:/opt/pyenv/shims:${PATH}"

# タイムゾーンの設定と基本的なパッケージのインストール
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        language-pack-ja \
        git \
        nano \
        vim \
        zip \
        unzip \
        wget \
        curl \
        tmux \
        make \
        build-essential \
        libssl-dev \
        zlib1g-dev \
        libbz2-dev \
        libreadline-dev \
        libsqlite3-dev \
        libvips \
        libvips-dev \
        libgl1-mesa-dev \
        screen \
        reptyr \
        software-properties-common && \
    # Google Cloud SDKのGPGキー追加 (apt-keyは非推奨になりつつある点に注意)
    wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add - && \
    apt-get update && \
    # 不要なキャッシュの削除
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# pyenv and Miniconda installation and settings
RUN curl -L https://github.com/pyenv/pyenv-installer/raw/master/bin/pyenv-installer | bash && \
    echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc && \
    echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc && \
    echo 'eval "$(pyenv init -)"' >> ~/.bashrc && \
    echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc && \
    $PYENV_ROOT/bin/pyenv install miniconda3-4.7.12 && \
    $PYENV_ROOT/bin/pyenv global miniconda3-4.7.12 && \
    conda update -n base -c defaults conda --yes && \
    conda install -n base -c defaults python=3.9.21 --yes && \
    conda clean -afy

# setting working directory
WORKDIR /workspace

COPY install.sh /workspace/install.sh
RUN chmod +x /workspace/install.sh && \
    /workspace/install.sh

# default command (start bash when container started)
CMD ["bash"]