

from datasets import load_dataset
import torch
import json
import os
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import AutoTokenizer
from transformers import TrainingArguments, Trainer


prefix = "Translate text to SQL: "
MAX_INPUT_LENGTH = 1000
MAX_TARGET_LENGTH = 1000
SCHEMA_PREFIX = "schema: "
QUESTION_PREFIX = "Translate text to SQL: "


def prepare_input(question, schema, query):
  '''Use three prefixes to join the question, schema, and query'''
  # f"{prefix} text: {question} db_id: {db_id} schema: {schema}

  input = f"{QUESTION_PREFIX} text: {question} {SCHEMA_PREFIX} {schema}"
  return input

def add_quotes(table, query_dict):
  # quotes query_dict["sel"]
  header = table["header"]
  query = query_dict["human_readable"]
  col_name = header[query_dict["sel"]]
  # print(col_name)
  result_query = query.replace(col_name, "'" + col_name + "'")
  
  # quotes query_dict["conds"]["column_index"] and quotes query_dict["conds"]["condision"]
  conds_col_idx_lst = query_dict["conds"]["column_index"]
  for cond_col_idx in conds_col_idx_lst:
    col_name = header[cond_col_idx]
    result_query = result_query.replace(col_name, "'" + col_name + "'")
  
  # add quotes to text values
  values = query_dict["conds"]["condition"]
  dtype_lst = table["types"]
  for idx in range(len(values)):
    dtype_idx = conds_col_idx_lst[idx]
    dtype = dtype_lst[dtype_idx]
    value = values[idx]
    if dtype == "text":
      result_query = result_query.replace(value, "'" + value + "'")
    else:
      try:
        float_value = float(value)
        # if successfully casted to float, don't need quote, do nothing
      except:
        # Add quote if can't be casted to float
        result_query = result_query.replace(value, "'" + value + "'")
    # if query_dict["agg"] != 0:

  return result_query
    
def process_wikisql(question, table, sql):
  '''Preprocess training data,
  don't use it for inference
  '''
  table_name = "table"
  num_tables = 1
  schema = ",".join(table["header"])
  schema = "num_tables: " + str(num_tables) + " (table_name: " + table_name + "; " + "table_cols: " + schema +")"

  quoted_sql = add_quotes(table, sql)
  
  return prepare_input(question, schema, quoted_sql)


def format_output(query):
    query = query["human_readable"]
    output = f"{query}"
    return output


# def get_query(output, queryToken = QUERY_PREFIX):
#     print(f"Raw output: {output}")
#     splitted = output.split("SELECT")
    
#     try:
#       output = "SELECT" + splitted[1]
#       output = output.strip()[:-len(END_TOKEN)]
#       output.replace(PAD_TOKEN, "")
#       return output.strip()
#     except IndexError:
#       return "No query"

# def infer(input, model, tokenizer, device):
#   inp = tokenizer(input, return_tensors="pt")
#   X = inp["input_ids"].to(device)
#   a = inp["attention_mask"].to(device)
#   output = model.generate(X, attention_mask=a , max_length=200)
#   output = tokenizer.decode(output[0])
#   return get_query(output)

# def infer_validation(data, model, tokenizer, device, pred_output_path):
#   pred_query = []
#   for i in range(len(data)):
#   # for i in range(50):
#     example = data[i]
#     question = example['question']
#     table = example['table']
#     query = example["sql"]["human_readable"]
#     # schema = ",".join(table["header"])
#     table_name = "table"
#     num_tables = 1
#     schema = ",".join(table["header"])
#     schema = "num_tables:" + str(num_tables) + " (table_name:" + table_name + "; " + "table_cols: " + schema +")"

#     input = f"<|startoftext|> {QUESTION_PREFIX} {question} {SCHEMA_PREFIX} {schema} {QUERY_PREFIX}"
#     print(f"Processed input: {input}")

#     generated_sql = infer(input, model, tokenizer, device)
#     pred_query.append(generated_sql)
#     print(f"Output query: {generated_sql}\n\n")
#     print(f"True query: {query}")

#   with open(pred_output_path, 'w') as fp:
#     for item in pred_query:
#         # write each item on a new line
#         try:
#           fp.write("%s\n" % item)
#         except UnicodeEncodeError:
#           try:
#             unicode_obj = item.encode('utf8')
#             fp.write("%s\n" % unicode_obj)
#           except:
#             fp.write("%s\n" % "ERROR Writing the query")
#   return

def main():
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  # Load Dataset
  print("Loading WikiSQL Dataset")
  dataset = load_dataset('wikisql')
  print("Spider Dataset Loaded")


  dataset_train = dataset["train"]
  dataset_test = dataset["test"]
  dataset_validation = dataset["validation"]

  model_checkpoint = 'google/flan-t5-base'
  print("Model_name: ", model_checkpoint)
  print("Loading Tokenizer...")
  tokenizer_path = os.path.join(os.getcwd(), "trained_tok")
  tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
  # tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-large')
  print("Tokenizer Custom Loaded...")
  print("max model length ", tokenizer.model_max_length)
  

  def preprocess_examples_wikisql(examples, tokenizer = tokenizer):
    # encode the question-query pairs
    questions = examples['question']
    queries = examples["sql"]
    tables = examples['table']
    inputs = [process_wikisql(questions[i],tables[i], queries[i]) for i in range(len(questions))]
    # outputs = [format_output(query) for query in queries]
    outputs = [add_quotes(table, query_dict)for query_dict, table in zip(queries, tables)]
    print("Input: ", inputs[0], "\nOutput: ", outputs[0])
    model_inputs = tokenizer(inputs, max_length=MAX_INPUT_LENGTH, truncation=True, padding='max_length', return_tensors="pt") 
    model_inputs['labels'] = tokenizer(outputs, max_length=MAX_TARGET_LENGTH, truncation=True, padding='max_length', return_tensors="pt").input_ids
    return model_inputs

  # small_train_dataset = dataset["train"].shuffle(seed=42).select(range(500))
  # small_val_dataset = dataset["validation"].shuffle(seed=42).select(range(100))

  print("Preprocess Train and Validation dataset...")
  dataset_train = dataset_train.map(preprocess_examples_wikisql, batched=True)
  dataset_validation = dataset_validation.map(preprocess_examples_wikisql, batched=True)
  print("Finished preprocessing data...")


  # Load T5 model
  print("Loading model...")
  model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large")
  model.to(device)
  model.resize_token_embeddings(len(tokenizer))
  print("Finished loading model...")
  

  # Train
  # output_dir = '/content/drive/MyDrive/Colab Notebooks/model_GPT2/wikisql'
  output_dir = os.getcwd()


  output_dir = os.getcwd()
  training_args = TrainingArguments(output_dir= output_dir, 
                                overwrite_output_dir=True,
                                evaluation_strategy = "epoch",
                                num_train_epochs=10,
                                learning_rate=1e-4,
                                per_device_train_batch_size=4,
                                per_device_eval_batch_size=4,
                                fp16=False,
                                warmup_steps=500, 
                                weight_decay=0.01)

  trainer = Trainer(
        model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=dataset_train,
        eval_dataset= dataset_validation,
    )
  print("Starting training...")
  trainer.train()
  print("Finished training...")
  print("Saving the model...")
  trainer.save_model()
  print("Model_saved...")

  # Load Trained Model
  print("Loading trained model...")
  fine_tuned_model = T5ForConditionalGeneration.from_pretrained(output_dir)
  fine_tuned_model.to(device)
  fine_tuned_model_tokenizer = AutoTokenizer.from_pretrained(output_dir)
  print("Model Loaded...")

  def remove_pad_token(query):
    query = query.lstrip('<pad>').strip()
    query = query.rstrip('</s>').strip()
    return query
  

  def infer(input, model, tokenizer):
    inp = tokenizer(input, return_tensors="pt")
    X = inp["input_ids"].to(device)
    a = inp["attention_mask"].to(device)
    output = model.generate(X, attention_mask=a , max_length=300)
    output = tokenizer.decode(output[0])
    return remove_pad_token(output)
  
  pred_output_path_validation = os.path.join(output_dir, "pred_test_data.txt")
  gold_output_path_validation = os.path.join(output_dir, "gold_test_data.txt")


  def model_output(data, pred_output_path, gold_output_path):
    predicted_queries = []
    gold_queries = []
    for i in range(len(data)):
        example = data[i]
        question = example['question']
        query = example["sql"]
        table = example['table']

        input = process_wikisql(question,table, query)
        print("Input: ", input)
        print("T Query: ",query)
        gold_queries.append(f"{query}")
        predicted_result = infer(input, fine_tuned_model, fine_tuned_model_tokenizer)
        print("P Query: ", predicted_result)
        print("")
        predicted_queries.append(predicted_result)

    print("Writing pred quereis out")
    with open(pred_output_path, 'w') as fp:
      for item in predicted_queries:
        # write each item on a new line
        fp.write("%s\n" % item.encode('utf-8'))

    print("Writing gold quereis out")
    with open(gold_output_path, 'w') as fp:
        for item in gold_queries:
            # write each item on a new line
            fp.write("%s\n" % item.encode('utf-8'))

    return

  print("Inferring trained model for validation set...")
  model_output(dataset_test, pred_output_path_validation, gold_output_path_validation)
  print("Finished inferring trained model for validation set...")
  print("ALL DONE!")


if __name__ == "__main__":
    main()