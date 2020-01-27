import tensorflow as tf
import json
from race_utils import RaceProcessor
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

