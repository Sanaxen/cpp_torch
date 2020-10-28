copy ..\..\..\..\..\libtorch\lib\*.dll .\bin\Release /v /y
copy C:\dev\opencv-3.4.0\build\x64\vc15\bin\*.dll .\bin\Release /v /y

copy ..\..\..\DCGAN\\dcgan\x64\Release\*.exe .\bin\Release /v /y

copy .\bin\Release\*.exe .\Application /v /y
