
copy .\x64\Release\*.exe run /v /y

cd run

rnn5.exe --x_dim 0 --y_dim 1 --lr 0.00001 --sequence_length 12 ^
--n_rnn 1 --hidden_size 256 --nfc 7 --fc_hidden_size 128 ^
--kNumberOfEpochs 10000 --input data/international-airline-passengers.csv --test 0.1 --prophecy 48 ^
--kLogInterval 50 --kTrainBatchSize 32 --use_gpu 0

pause
cd ..
