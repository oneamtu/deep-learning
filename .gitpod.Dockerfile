FROM gitpod/workspace-python

RUN pyenv install 3.8 \
    && pyenv global 3.8