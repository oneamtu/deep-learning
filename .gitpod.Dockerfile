FROM gitpod/workspace-python

RUN pyenv install 3.8 \
    && pyenv global 3.8

RUN sudo apt-get update && sudo apt-get -y install libgles2-mesa-dev 