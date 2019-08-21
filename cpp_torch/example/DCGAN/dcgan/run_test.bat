
copy .\x64\Release\*.exe run /v /y

cd run
dcgan_generate_test.exe

cd ..
