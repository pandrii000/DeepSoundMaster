for (( data_start=0; data_start<=320; data_start+=20 ))
do
    finish_part=$((data_start + 20))
    python ./train_model.py "$data_start-$finish_part"
done
