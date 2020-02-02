
## create google status bucket with name
```
bertqwert
```

## start CTPU
```
ctpu up -tf-version=1.15 -tpu-size=v3-8
```
* We need tf == 1.15.0 and memory > 8 G
* See: https://cloud.google.com/tpu/pricing

## Install
```
# Run pip install --upgrade pip if tensorflow 1.15 cannot be found

# CPU Version of TensorFlow
pip install tensorflow==1.15   
pip install tensorflow_hub==0.7

# tensorflow-gpu==1.15  # GPU version of TensorFlow

pip install sentencepiece

# Install six
pip install six==1.12.0
```

## Setup

```
# Install Albert
git clone https://github.com/google-research/ALBERT ALBERT

# Fetch and unzip RACE dataset
gsutil cp gs://bertqwert/dataset/RACE.tar.gz . &&  tar xvf RACE.tar.gz &&  rm RACE.tar.gz

# Compine all the example files to one single file
for f in RACE/train/middle/*.txt; do (cat "${f}"; echo) >> RACE/train/middle/all.txt; done
for f in RACE/train/high/*.txt; do (cat "${f}"; echo) >> RACE/train/high/all.txt; done
for f in RACE/test/middle/*.txt; do (cat "${f}"; echo) >> RACE/test/middle/all.txt; done
for f in RACE/test/high/*.txt; do (cat "${f}"; echo) >> RACE/test/high/all.txt; done
for f in RACE/dev/middle/*.txt; do (cat "${f}"; echo) >> RACE/dev/middle/all.txt; done
for f in RACE/dev/high/*.txt; do (cat "${f}"; echo) >> RACE/dev/high/all.txt; done

# Copy the files to bucket
gsutil cp ./RACE/train/middle/all.txt gs://bertqwert/RACE/train/middle/all.txt
gsutil cp ./RACE/train/high/all.txt gs://bertqwert/RACE/train/high/all.txt
gsutil cp ./RACE/test/middle/all.txt gs://bertqwert/RACE/test/middle/all.txt
gsutil cp ./RACE/test/high/all.txt gs://bertqwert/RACE/test/high/all.txt
gsutil cp ./RACE/dev/middle/all.txt gs://bertqwert/RACE/dev/middle/all.txt
gsutil cp ./RACE/dev/high/all.txt gs://bertqwert/RACE/dev/high/all.txt


# Fetch and unzip albert pre-trained model file
gsutil cp gs://bertqwert/dataset/albert_xxlarge_v2.tar.gz . &&  tar xvf albert_xxlarge_v2.tar.gz  &&  rm albert_xxlarge_v2.tar.gz

# Copy the folder to bucket
gsutil cp -r albert_xxlarge_v2 gs://bertqwert

# Notes: don't delet the folder albert_xxlarge_v2 because the file: ./albert_xxlarge/30k-clean.model is required by local.
```
gsutil cp gs://bertqwert/dataset/albert_xxlarge/30k-clean.model ./albert_xxlarge/30k-clean.model
```

* The trained model cannot be used by other cpu, gpu and other tpu device (config), you only can do prediction on the same tpu environment (that is ridiculous). Someone proposes this solution with `estimator = tf.contrib.tpu.TPUEstimator(..., export_to_tpu=False)` at https://github.com/tensorflow/tensorflow/issues/25652. But it didn't work for me.

## run 
* When running under tpu mode, make sure all input, model and out folder/files in gs.
* The file: spm_model_file should be in local
```
python ./ALBERT/run_race.py \
  --albert_config_file=gs://bertqwert/albert_xxlarge/albert_config.json \
  --output_dir=gs://bertqwert/out \
  --train_file=gs://bertqwert/train_file.d \
  --eval_file=gs://bertqwert/eval_file.d \
  --data_dir=gs://bertqwert \
  --init_checkpoint=gs://bertqwert/albert_xxlarge/model.ckpt-best \
  --spm_model_file=./albert_xxlarge/30k-clean.model \
  --max_seq_length=512 \
  --max_qa_length=128 \
  --do_train \
  --do_eval \
  --train_batch_size=32 \
  --eval_batch_size=8 \
  --learning_rate=1e-5 \
  --train_step=12000 \
  --warmup_step=1000 \
  --save_checkpoints_steps=100 \
  --use_tpu \
  --tpu_name=simonrc202001
  ```