#!/bin/bash
now=$(date +"%r")
echo "Started cron job at: $now"
cd /home/naresh/flask_app
/root/anaconda3/bin/python /home/naresh/flask_app/cronFile.py
now=$(date +"%r")
echo "Successfully finished cron job at: $now"