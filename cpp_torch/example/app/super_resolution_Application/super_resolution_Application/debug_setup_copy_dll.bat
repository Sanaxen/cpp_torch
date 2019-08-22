copy ..\..\..\..\..\libtorch\lib\*.dll .\bin\Debug /v /y
copy C:\dev\opencv-3.4.0\build\x64\vc15\bin\*.dll .\bin\Debug /v /y

copy .\pretrained_model\*.* .\bin\Debug /v /y
copy ..\..\..\SUPER_RESOLUTION_ESPCN\\super_resolution_espcn\x64\Release\*_test.exe .\bin\Debug /v /y