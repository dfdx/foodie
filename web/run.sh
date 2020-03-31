#!/bin/bash
sudo su
cd ~ubuntu/work/foodie/web
export FLASK_APP=main.py
/home/ubuntu/work/foodie/venv/bin/flask run --host 0.0.0.0 --port 80
