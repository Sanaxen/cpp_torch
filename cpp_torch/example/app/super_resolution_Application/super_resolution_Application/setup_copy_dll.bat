copy ..\..\..\..\..\libtorch\lib\*.dll Application /v /y
copy C:\dev\opencv-3.4.0\build\x64\vc15\bin\*.dll Application /v /y

copy .\bin\Release\super_resolution_Application*.* Application /v /y
copy .\pretrained_model\*.* Application /v /y

copy ..\..\..\SUPER_RESOLUTION_ESPCN\\super_resolution_espcn\x64\Release\*.exe Application /v /y