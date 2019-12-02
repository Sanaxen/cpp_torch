
copy .\x64\Release\*.exe run /v /y

cd run
rnn5.exe --x_dim 1 --y_dim 1 --explanatory_variable 1 --n_rnn 1 --input data/sample1.csv ^
--solver Adam --lr 0.001  --kTrainBatchSize 128 --kNumberOfEpochs 90 --kLogInterval 50 --use_gpu 0

cd ..
