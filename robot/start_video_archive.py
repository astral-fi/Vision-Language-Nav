"""Serve devbox recordings for viewing without copying videos to the laptop."""
from pathlib import Path
import subprocess
import urllib.request

runtime = Path(__file__).resolve().parent
try:
    urllib.request.urlopen('http://127.0.0.1:8768/', timeout=2).close()
    print('Video archive already available on 127.0.0.1:8768')
except OSError:
    with (runtime / 'logs/video_archive.log').open('ab', buffering=0) as log:
        process = subprocess.Popen(['python3', '-m', 'http.server', '8768', '--bind', '127.0.0.1',
                                    '--directory', str(runtime / 'runs')],
                                   stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
    (runtime / 'logs/video_archive.pid').write_text(str(process.pid) + '\n')
    print('Video archive PID', process.pid)
