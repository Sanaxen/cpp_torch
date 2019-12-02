
copy .\x64\Release\*.exe run /v /y

cd run

rnn5.exe --x_dim 1 --y_dim 1 --lr 0.0001 --sequence_length 120 --out_sequence_length 6 ^
--n_rnn 1 --hidden_size 64 --nfc 6 --fc_hidden_size 32 ^
--kNumberOfEpochs 10000 --input data/example_wp_log_peyton_manning.csv --test 0.3 --prophecy 48 ^
--kLogInterval 10 --kTrainBatchSize 32 --use_gpu 0

pause
cd ..
