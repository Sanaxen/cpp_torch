set PYPATH=%USERPROFILE%\Anaconda3

copy .\x64\Release\*.exe run /v /y

cd run
torch_jit.exe

cd ..
pause