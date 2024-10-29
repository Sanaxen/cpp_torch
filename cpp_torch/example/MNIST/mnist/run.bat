copy ..\..\..\..\libtorch\lib\*.dll run /v /y
copy ..\..\..\example\mnist\mnist\run\data\* run\data /v /y

copy .\x64\Release\*.exe run /v /y

cd run
mnist.exe

cd ..
:del /Q .\run\*.dll
