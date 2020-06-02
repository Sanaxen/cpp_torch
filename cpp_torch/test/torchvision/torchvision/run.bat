set PYPATH=%USERPROFILE%\Anaconda3

copy .\x64\Release\*.exe run /v /y

cd run
torchvision.exe

cd ..
