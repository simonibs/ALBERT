import tensorflow as tf
# Enable the eager mode so we can explore the tfrecord
tf.enable_eager_execution()
import json
import os
from race_utils import RaceProcessor
import race_utils
import fine_tuning_utils
import classifier_utils
import modeling
from tensorflow.contrib import cluster_resolver as contrib_cluster_resolver
from tensorflow.contrib import tpu as contrib_tpu

processor = RaceProcessor(True, True, True, True)

file_path = "./data/RACE_test_middle_all.txt"
i = 1
cur_data = ""
with tf.gfile.Open(file_path) as f:
  for line in f:
    cur_data = json.loads(line.strip())
    if 1 == 1:
      break
# Print Raw data      
print(cur_data)

example_id = cur_data["id"]
article = cur_data["article"]
question = cur_data["questions"][0]
options = cur_data["options"][0]
answer = cur_data["answers"][0]
example = processor.get_example(example_id, article, question, options, answer)
# Print example
print(example)

label_list = ["A", "B", "C", "D"]

# Build a tokenizer object
tokenizer = fine_tuning_utils.create_vocab(
    vocab_file=None,
    do_lower_case=False,
    spm_model_file="./data/albert_xxlarge_30k-clean.model",
    hub_module=None)

text = tokenizer.tokenize("this is a good test.")
print(text)

predict_file =  "./out/predict.tf_record"
predict_examples = [example, example, example,example,example]
max_seq_length = 512
max_qa_length = 128
race_utils.file_based_convert_examples_to_features(
    predict_examples, label_list,
    max_seq_length, tokenizer,
    predict_file, max_qa_length)

# Explore the Example tf record
# filenames = ["./out/predict.tf_record"]
# raw_dataset = tf.data.TFRecordDataset(filenames)
# for raw_record in raw_dataset.take(1):
#   example = tf.train.Example()
#   example.ParseFromString(raw_record.numpy())
#   print(example)

# Create input fn
predict_drop_remainder = False
predict_batch_size = 8
task_name = "race"
predict_input_fn = classifier_utils.file_based_input_fn_builder(
    input_file=predict_file,
    seq_length=max_seq_length,
    is_training=False,
    drop_remainder=False,
    task_name=task_name,
    use_tpu=False,
    bsz=predict_batch_size,
    multiple=len(label_list))

checkpoint_path = "./data/out_model.ckpt-best"
predict_steps = 5

albert_config = modeling.AlbertConfig.from_json_file("./data/albert_config.json")

model_fn = race_utils.model_fn_builder(
      albert_config=albert_config,
      num_labels=len(label_list),
      init_checkpoint=None,
      learning_rate=1e-5,
      num_train_steps=12000,
      num_warmup_steps=1000,
      use_tpu=False,
      use_one_hot_embeddings=False,
      max_seq_length=max_seq_length,
      dropout_prob=0.1,
      hub_module=None)

# If TPU is not available, this will fall back to normal Estimator on CPU
# or GPU.
tpu_cluster_resolver = None
run_config = contrib_tpu.RunConfig(
    cluster=tpu_cluster_resolver,
    master=None,
    model_dir="./out",
    save_checkpoints_steps=int(100),
    keep_checkpoint_max=0,
    tpu_config=contrib_tpu.TPUConfig(
        iterations_per_loop=1000,
        num_shards=8,
        per_host_input_for_training=contrib_tpu.InputPipelineConfig.PER_HOST_V2))

estimator = contrib_tpu.TPUEstimator(
    use_tpu=False,
    model_fn=model_fn,
    config=run_config,
    train_batch_size=32,
    eval_batch_size=8,
    predict_batch_size=8)

result = estimator.evaluate(
    input_fn=predict_input_fn,
    steps=predict_steps,
    checkpoint_path=checkpoint_path)

output_predict_file = "./out/predict_results.txt"
with tf.gfile.GFile(output_predict_file, "w") as pred_writer:
  # num_written_lines = 0
  tf.logging.info("***** Predict results *****")
  pred_writer.write("***** Predict results *****\n")
  for key in sorted(result.keys()):
    tf.logging.info("  %s = %s", key, str(result[key]))
    pred_writer.write("%s = %s\n" % (key, str(result[key])))
  # pred_writer.write("best = {}\n".format(best_perf))

print(result)