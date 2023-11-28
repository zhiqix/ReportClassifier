from config import *
from utils import *
from model import *
from transformers import AutoTokenizer
import os
import torch
import time


def find_all_files_in_directory(dir_path: str) -> list:
    all_files = []

    for root, dirs, files in os.walk(dir_path):
        for file in files:
            all_files.append(os.path.join(root, file))

    return all_files


def load_report(report_name):
    tokenizer = AutoTokenizer.from_pretrained(BGE_MODEL)

    with open(report_name) as f:
        lines = f.readlines()

    sentences = []
    for line in lines:
        # Filter out empty strings
        sentences.extend([sent for sent in line.split('。') if sent])

    input_data = []
    masks = []
    origin_input = []
    for sentence in sentences:
        # Tokenization & Add special tokens
        tokened = tokenizer(sentence, max_length=TEXT_LEN, padding=True, truncation=True)
        input_ids = tokened['input_ids']
        mask = tokened['attention_mask']
        # Pad the input_ids to ensure that the length of the input_ids reaches TEXT_LEN.
        if len(input_ids) < TEXT_LEN:
            pad_len = (TEXT_LEN - len(input_ids))
            input_ids += [BGE_PAD_ID] * pad_len
            mask += [0] * pad_len

        input_data.append(input_ids[:TEXT_LEN])
        masks.append(mask[:TEXT_LEN])
        origin_input.append(sentence)

    return torch.tensor(input_data), torch.tensor(masks), origin_input


if __name__ == '__main__':
    # Record the time when the code execution starts.
    start_time = time.time()

    file_list = find_all_files_in_directory(SOURCE_DIR)
    print("Number of files in the directory: ", len(file_list))

    # Record progress.
    cnt = 0

    model = torch.load(TRAINED_MODEL, map_location=DEVICE)
    model.eval()

    for file in file_list:
        # Calculate the file size, omitting files with a size of 0.
        size = os.path.getsize(file)
        if size == 0:
            cnt += 1
            print("Empty file.")
            continue

        # Extract the complete name of the file (including both the filename and the extension) from the file path.
        file_name = os.path.basename(file)
        # Separate the filename and the extension.
        file_name_without_extension, file_extension = os.path.splitext(file_name)

        input, mask, origin_input = load_report(file)
        y_pred = []

        with torch.no_grad():
            # Chunking to prevent OutOfMemoryError when processing long text sequences.
            sequence_length = 440
            input_sequences = torch.split(input, sequence_length, dim=0)
            mask_sequences = torch.split(mask, sequence_length, dim=0)

            for i in range(len(input_sequences)):
                input_chunk = input_sequences[i]
                mask_chunk = mask_sequences[i]
                input_chunk = input_chunk.to(DEVICE)
                mask_chunk = mask_chunk.to(DEVICE)
                test_pred = model(input_chunk, mask_chunk)
                predicted_classes = torch.argmax(test_pred, dim=1)
                y_pred += predicted_classes.data.tolist()

        torch.cuda.empty_cache()

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(f'{OUTPUT_DIR}/{file_name_without_extension}_8.txt', 'w', newline='') as f1:
            for i in range(len(input)):
                # Filter short texts.
                if len(origin_input[i]) >= 10:
                    # Extract data with label=8.
                    if (y_pred[i]) == 8:
                        if origin_input[i].endswith('\n'):
                            f1.write(origin_input[i])
                        else:
                            f1.write(origin_input[i])
                            f1.write('\n')

        cnt += 1
        # Record process.
        if cnt % 35 == 0:
            print("Elapsed time: ", (time.time() - start_time) / 60 / 60, "h", '\t', 'Progress:', cnt, "/",
                  len(file_list))

    # Record the time when the code execution finishes.
    end_time = time.time()

    # Calculate the execution time of the code (in seconds).
    execution_time = end_time - start_time
    print(f"The code execution time is:  {execution_time:.2f} seconds")
