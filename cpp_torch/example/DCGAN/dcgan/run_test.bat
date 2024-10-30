copy ..\..\..\..\libtorch\lib\*.dll run /v /y
copy ..\..\..\..\libtorch\torchvision\bin\*.dll run /v /y
copy C:\dev\opencv\build\x64\vc16\bin\opencv_world4100.dll run /v /y

copy .\x64\Release\*.exe run /v /y

cd run
dcgan_generate_test.exe

cd ..
