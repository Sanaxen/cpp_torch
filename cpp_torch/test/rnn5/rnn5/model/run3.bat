
copy .\x64\Release\*.exe run /v /y

cd run

rnn5.exe --x_dim 0 --y_dim 1 --lr 0.00001 --sequence_length 12 ^
--n_rnn 1 --hidden_size 128 --nfc 5 --fc_hidden_size 128 ^
--kNumberOfEpochs 10000 --input data/co2-ppm-mauna-loa-19651980.csv --test 0.3 --prophecy 48 ^
--kLogInterval 50 --kTrainBatchSize 32

pause
cd ..
