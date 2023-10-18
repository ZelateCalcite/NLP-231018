import torch
import argparse
import json
import logging
import os
import random

from tqdm import tqdm
import jsonlines

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
LOGGER = logging.getLogger(__name__)


def get_json_lines(inp_file):
    lines = []
    with jsonlines.open(inp_file) as reader:
        for obj in reader:
            lines.append(obj)
    return lines


def read_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str)
    parser.add_argument("--output_folder", default='outputs/', type=str)
    parser.add_argument("--output_prefix", default='', type=str)
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def get_outputs(args, model, tokenizer, batch):
    texts = [x["prompt"] for x in batch]
    encoding = tokenizer(texts, return_tensors='pt', truncation=True, padding="max_length",
                         max_length=tokenizer.model_max_length).to(model.device)
    with torch.no_grad():
        if args.nosample:
            generated_ids = model.generate(**encoding, max_length=100, do_sample=False, repetition_penalty=1.2)
        else:
            generated_ids = model.generate(**encoding, max_length=100, do_sample=True, top_p=0.9, top_k=0,
                                           repetition_penalty=1.2, temperature=0.7)
    generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    return generated_texts


def get_outputs_bytype(args, model, tokenizer, batch, decoding_type):
    texts = [x["prompt"] for x in batch]
    encoding = tokenizer(texts, return_tensors='pt', truncation=True, padding="max_length",
                         max_length=tokenizer.model_max_length).to(model.device)
    with torch.no_grad():
        if decoding_type == 'greedy':
            generated_ids = model.generate(**encoding, max_length=100, do_sample=False, repetition_penalty=1.2,
                                           decoder_start_token_id=model.config.bos_token_id)
        else:
            generated_ids = model.generate(**encoding, max_length=200, do_sample=True, top_p=0.9, top_k=0,
                                           repetition_penalty=1.2, temperature=0.7,
                                           decoder_start_token_id=model.config.bos_token_id)
    generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return generated_texts


def run_generation(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device', device)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    model = model.to(device)
    model_name_string = args.model
    if model_name_string[-1] == '/':
        model_name_string = model_name_string[:-1]
    if 'checkpoint' in model_name_string:
        model_name_string = model_name_string.split('/')[-2] + '--' + model_name_string.split('/')[-1]
    else:
        model_name_string = model_name_string.split('/')[-1]
    data_json_lines = get_json_lines(args.input_file)

    greedy_decode_data = []
    sample_decode_data = []
    greedy_decode_tasks = set()
    sample_decode_tasks = set()
    for i, dp in enumerate(data_json_lines):
        if 'generation' in dp['task'] or 'fill' in dp['task'] or 'summarization' in dp['task']:
            dp['decoding_type'] = 'sample'
            sample_decode_tasks.add(dp['task'])
            sample_decode_data.append(dp)
        else:
            dp['decoding_type'] = 'greedy'
            greedy_decode_tasks.add(dp['task'])
            greedy_decode_data.append(dp)

    print('greedy_decode_tasks', greedy_decode_tasks)
    print('sample_decode_tasks', sample_decode_tasks)

    input_file_name = args.input_file.split('/')[-1]

    os.makedirs(args.output_folder, exist_ok=True)
    with open(os.path.join(args.output_folder, args.output_prefix + "_" + model_name_string + "-" + input_file_name),
              'w') as outfile:
        batch_chunks_greedy = chunks(greedy_decode_data, args.batch_size)
        for b, batch in tqdm(enumerate(batch_chunks_greedy), total=len(greedy_decode_data) // args.batch_size):
            outputs = get_outputs_bytype(args, model, tokenizer, batch, decoding_type='greedy')
            for d, dp in enumerate(batch):
                batch[d]['output'] = outputs[d]
                if b % 100 == 0:
                    print(batch[d]['output'], ' -- ', batch[d]['all_outputs'])
                json.dump(batch[d], outfile)
                outfile.write('\n')
                outfile.flush()
        args.batch_size -= 10
        batch_chunks_sample = chunks(sample_decode_data, args.batch_size)
        for b, batch in tqdm(enumerate(batch_chunks_sample), total=len(sample_decode_data) // args.batch_size):
            outputs = get_outputs_bytype(args, model, tokenizer, batch, decoding_type='sample')
            for d, dp in enumerate(batch):
                batch[d]['output'] = outputs[d]
                if b % 100 == 0:
                    print(batch[d]['output'], ' -- ', batch[d]['all_outputs'])
                json.dump(batch[d], outfile)
                outfile.write('\n')
                outfile.flush()


if __name__ == "__main__":
    args = read_args()
    print(args)
    random.seed(args.seed)
    run_generation(args)
