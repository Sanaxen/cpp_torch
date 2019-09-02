
copy .\x64\Release\*.exe run /v /y

cd run
del /Q *.pt
dcgan.exe

cd ..
