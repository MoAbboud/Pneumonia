web: cd PneumoniaDetectorWebApp && gunicorn main:app --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 180 --worker-class=gthread --worker-tmp-dir /dev/shm
