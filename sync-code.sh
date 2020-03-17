#!/bin/bash
rsync -zhav --exclude venv . foodie:work/foodie/
