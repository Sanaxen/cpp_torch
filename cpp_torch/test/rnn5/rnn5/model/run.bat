
copy .\x64\Release\*.exe run /v /y

cd run

rnn5.exe --x_dim 1 --y_dim 3 --lr 0.0001 --sequence_length 10 ^
--n_rnn 1 --hidden_size 64 --nfc 6 --fc_hidden_size 32 ^
--kNumberOfEpochs 2000 --input data/sample.csv --test 0.3 --prophecy 2000 ^
--kLogInterval 50 --kTrainBatchSize 48  --normalize 2 --use_gpu 0

pause
cd ..
