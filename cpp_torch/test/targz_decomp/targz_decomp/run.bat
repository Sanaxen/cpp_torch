copy test.tar.gz run\data /v /y
copy test.tar.zip run\data /v /y
copy .\x64\Release\*.exe run /v /y

cd run
targz_decomp.exe

cd ..
