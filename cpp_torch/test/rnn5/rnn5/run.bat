
copy .\x64\Release\*.exe run /v /y

cd run

rnn5.exe --x_dim 1 --y_dim 3 --lr 0.00001 --sequence_length 20 ^
--n_rnn 1 --hidden_size 128 --nfc 5 --fc_hidden_size 128 ^
--kNumberOfEpochs 2000 --input data/sample.csv --test 0.3 --prophecy 2000 ^
--kLogInterval 50 --kTrainBatchSize 32

pause
cd ..
