set PYPATH=%USERPROFILE%\Anaconda3

copy .\x64\Release\*.exe run /v /y

cd run

Autograd.exe

cd ..
