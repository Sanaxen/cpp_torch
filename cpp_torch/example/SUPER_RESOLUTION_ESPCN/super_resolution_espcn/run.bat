copy ..\..\..\..\libtorch\lib\*.dll run /v /y


copy .\x64\Release\*.exe run /v /y

cd run
super_resolution_espcn.exe

cd ..
del /Q .\run\*.dll
