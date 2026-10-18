@echo off
setlocal
set URL=https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe
set OUT=C:\temp\python-3.12.10-amd64.exe
mkdir C:\temp 2>nul
powershell -NoProfile -Command "Invoke-WebRequest -Uri '%URL%' -OutFile '%OUT%'; Start-Process -FilePath '%OUT%' -ArgumentList '/quiet InstallAllUsers=0 PrependPath=1' -Wait"
