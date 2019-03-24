#!/bin/bash

# Script to start the Jupyter Notebook
# Author: Cristiano Fraga G. Nunes
# E-mail: cfgnunes@gmail.com
# Needs: screen

set -eu

screen -d -m -S bukkit bash -c "cd .. && source .venv/bin/activate && jupyter notebook --no-browser --ip='0.0.0.0'"
