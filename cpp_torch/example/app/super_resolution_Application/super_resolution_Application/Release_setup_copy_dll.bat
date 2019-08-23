copy ..\..\..\..\..\libtorch\lib\*.dll .\bin\Release /v /y
copy C:\dev\opencv-3.4.0\build\x64\vc15\bin\*.dll .\bin\Release /v /y

copy .\pretrained_model\*.* .\bin\Release /v /y
copy ..\..\..\SUPER_RESOLUTION_ESPCN\\super_resolution_espcn\x64\Release\*.exe .\bin\Release /v /y
copy ..\..\..\SUPER_RESOLUTION_ESPCN\\super_resolution_espcn\run\*.pt .\bin\Release /v /y

