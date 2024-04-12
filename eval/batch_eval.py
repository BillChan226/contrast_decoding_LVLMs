import os
import re
import subprocess
import csv
import argparse

# Set the directory where the chair.json files are located
# directory = './paper_result/32_tokens/minigpt4/'

parser = argparse.ArgumentParser(description="POPE-Adv evaluation on LVLMs.")

parser.add_argument(
    "-c",
    "--chair-path",
    type=str,
    required=True,
    help="Path to the generated CHAIR captions",
)
parser.add_argument(
    "-e",
    "--evaluator",
    type=str,
    required=True,
    help="chair or pope",
)
parser.add_argument(
    "-o",
    "--chair_only",
    action="store_true",
    help="evaluate chair only (not the generation quality scores)",
)
parser.add_argument(
    "-p",
    "--pope_type",
    type=str,
    required=False,
    help="pope type",
)
parser.add_argument(
    "--coco_path",
    type=str,
    default="eval_dataset/val2014/annotations",
    required=False,
    help="path to coco dataset",
)

args = parser.parse_known_args()[0]

directory = args.chair_path
evaluator = args.evaluator
pope_type = args.pope_type
chair_only = args.chair_only
coco_path = args.coco_path

# Function to run the eval_hallucination command and parse the output
def run_eval_chair(file_path, chair_only=False):
    # Running the eval_hallucination command for the given file
    if chair_only == True:
        result = subprocess.run(
            ["python", "eval/chair.py", "--cap_file", file_path, "--coco_path", coco_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        metrics = re.findall(
            r"\d+\.\d+",
            result.stdout.split("ground truth captions")[-1].split("CHAIR results")[0],
        )

        metrics = metrics if metrics else None
    
    else:
        result = subprocess.run(
            ["python", "eval/eval_hallucination.py", "--chair_input_path", file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # print("result", result)

        metrics = re.findall(
            r"\d+\.\d+",
            result.stdout.split("ground truth captions")[-1].split("CHAIR results")[0],
        )


        # print("metrics", metrics)
        # input()

        # print("result.stdout", result.stdout)

        metrics = metrics if metrics else None

        # Regex patterns to find hallucinate_sum and object_sum
        hallucination_sum_pattern = r"hallucinate_sum:\s+(\d+)"
        object_sum_pattern = r"object_sum:\s+(\d+)"
        words_sum_pattern = r"words_sum:\s+(\d+)"
        bleu_pattern = r"bleu:\s+(\d+\.\d+)"
        log_cider_pattern = r"log_cider:\s+(\d+\.\d+)"

        rest_of_stdout = result.stdout.split("meteor")[-1]
        # print("rest_of_stdout", rest_of_stdout)
        # Extracting values using regex
        hallucination_sum_match = re.search(hallucination_sum_pattern, rest_of_stdout)
        object_sum_match = re.search(object_sum_pattern, rest_of_stdout)
        words_sum_match = re.search(words_sum_pattern, rest_of_stdout)
        bleu_match = re.search(bleu_pattern, rest_of_stdout)
        log_cider_match = re.search(log_cider_pattern, rest_of_stdout)

        # # Extracted values
        # hallucination_sum = int(hallucination_sum_match.group(1)) if hallucination_sum_match else None
        # metrics.append(str(hallucination_sum))
        # object_sum = int(object_sum_match.group(1)) if object_sum_match else None
        # metrics.append(str(object_sum))
        # words_sum = int(words_sum_match.group(1)) if words_sum_match else None
        # metrics.append(str(words_sum))
        bleu = float(bleu_match.group(1)) if bleu_match else None
        metrics.append(str(bleu)[:5])
        log_cider = float(log_cider_match.group(1)) if log_cider_match else None
        metrics.append(str(log_cider)[:5])

        # print("metrics", metrics)

        # Returning the extracted metrics
    return metrics

def run_eval_pope(file_path, pope_type):

    result = subprocess.run(
        ["python", "eval/opope_eval.py", "-c", file_path, "--pope_type", pope_type],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    pattern = r"(Accuracy|Precision|Recall|F1 score|Yes ratio):\s*([\d.]+)"
    
    # Find all matches
    matches = re.findall(pattern, result.stdout)

    # Convert the matches to a dictionary
    metrics = {metric: float(value)*100 for metric, value in matches}
    try:
        metrics = [str(metrics[metric])[:6] for metric in ["Accuracy", "Precision", "Recall", "F1 score", "Yes ratio"]]
    except:
        print("Error in metrics", metrics)
    return metrics


def extract_info_from_filename(filename):
    # Corrected regex pattern to accurately capture 'llava-1.5' as the model name
    if "skip" not in filename:
        match = re.search(
            # r"([^_]+)_([^_]+)_([^_]+)_box_([\d.]+)_beams_(\d+)_k_(\d+)_coco_expand_ratio_([\d.]+)_seed_(\d+)_max_tokens_(\d+)_samples_(\d+)_chair",
            r"([^_]+)_([^_]+)_([^_]+)_box_([\d.]+)_beams_(\d+)_k_(\d+)_coco_expand_ratio_([\d.]+)_seed_(\d+)_max_tokens_(\d+)_samples_(\d+)_",
            filename,
        )
    else:
        match = re.search(
            r"([^_]+)_([^_]+)_([^_]+)_box_([\d.]+)_beams_(\d+)_k_(\d+)_coco_expand_ratio_([\d.]+)_seed_(\d+)_max_tokens_(\d+)_samples_(\d+)_skip_(\d+)_",
            filename,
        )
    if match:
        # print(match.groups())
        return match.groups()
    else:
        return "-", "-", "-", -1, -1, -1, -1, -1, -1, -1, -1

if evaluator == "chair":
    # Initialize the markdown table with headers
    markdown_table = "| Backbone | Decoder | Detector | Box | Ratio | Beam | K | Seed | SPICE | METEOR | CIDEr | CHAIRs | CHAIRi | BLEU | Log CIDEr |  Num of Samples | Max Tokens |\n"
    markdown_table += "|---------|---------|---------|---------|-----------|-----------|----------|------------|-------|--------|-------|-------|-------|--------|--------|--------|--------|\n"

    # Prepare the CSV file
    csv_file_path = "eval/eval_chair_results.csv"
    csv_columns = [
        "Backbone",
        "Decoder",
        "Detector",
        "Box",
        "Ratio",
        "Beam",
        "K",
        "Seed",
        "SPICE",
        "METEOR",
        "CIDEr",
        "CHAIRs",
        "CHAIRi",
        "BLEU",
        "Log CIDEr",
        "Max Tokens",
        "Num of Samples",
        # "Skip Samples"
    ]
elif evaluator == "pope":
    # Initialize the markdown table with headers
    markdown_table = "| Backbone | Decoder | Detector | Box | Ratio | Beam | K | Seed | Accuracy | Precision | Recall | F1 score | Yes ratio | Max Tokens | Num of Samples | Skip Samples |\n"
    markdown_table += "|---------|---------|---------|---------|-----------|-----------|----------|------------|-------|--------|-------|-------|--------|--------|--------|--------|\n"

    # Prepare the CSV file
    csv_file_path = "eval/eval_pope_results.csv"
    csv_columns = [
        "Backbone",
        "Decoder",
        "Detector",
        "Box",
        "Ratio",
        "Beam",
        "K",
        "Seed",
        "Accuracy",
        "Precision",
        "Recall",
        "F1 score",
        "Yes ratio",
        "Max Tokens",
        "Num of Samples",
        "Skip Samples"
    ]

# Start writing to the CSV file
with open(csv_file_path, "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
    writer.writeheader()  # write the header

    file_names = os.listdir(directory)
    # Sort the file names by beam size, then by k number, then by seed number
    sorted_file_names = sorted(
        file_names, key=lambda name: extract_info_from_filename(name)[1:]
    )

    
    # Loop through each file in the directory and process it
    for file_name in sorted_file_names:
        if evaluator == "chair" and chair_only == False:
            if file_name.endswith("_chair.json"):
                file_path = os.path.join(directory, file_name)
                print(file_path)
                # Extract information from filename
                # decoder,  beam_size, k_number, expand_ratio, seed_number = extract_info_from_filename(file_name)
                (
                    backbone,
                    decoder,
                    detector,
                    box,
                    beam_size,
                    k_number,
                    expand_ratio,
                    seed_number,
                    max_tokens,
                    num_samples,
                    skip_samples
                ) = extract_info_from_filename(file_name)
        
                metrics = run_eval_chair(file_path)

                print(
                f"| {backbone} | {decoder} | {detector} | {box} | {expand_ratio} | {beam_size} | {k_number} | {seed_number} | {' | '.join(metrics)} | {max_tokens} | {num_samples} |\n"
                )
            else:
                continue
        elif evaluator == "chair" and chair_only == True:
            if file_name.endswith("_generated_captions.json"):
                file_path = os.path.join(directory, file_name)
                print(file_path)
                # Extract information from filename
                # decoder,  beam_size, k_number, expand_ratio, seed_number = extract_info_from_filename(file_name)
                (
                    backbone,
                    decoder,
                    detector,
                    box,
                    beam_size,
                    k_number,
                    expand_ratio,
                    seed_number,
                    max_tokens,
                    num_samples,
                    skip_samples
                ) = extract_info_from_filename(file_name)
        
                metrics = run_eval_chair(file_path, True)

                print(
                f"| {backbone} | {decoder} | {detector} | {box} | {expand_ratio} | {beam_size} | {k_number} | {seed_number} | {' | '.join(metrics)} | {max_tokens} | {num_samples} |\n"
                )
            else:
                continue

        elif evaluator == "pope":
            
            if file_name.endswith("_generated_captions.json"):
                file_path = os.path.join(directory, file_name)
                print(file_path)
                # Extract information from filename
                # decoder,  beam_size, k_number, expand_ratio, seed_number = extract_info_from_filename(file_name)
                (
                    backbone,
                    decoder,
                    detector,
                    box,
                    beam_size,
                    k_number,
                    expand_ratio,
                    seed_number,
                    max_tokens,
                    num_samples,
                    skip_samples,
                ) = extract_info_from_filename(file_name)

                metrics = run_eval_pope(file_path, pope_type)

                print(
                    f"| {backbone} | {decoder} | {detector} | {box} | {expand_ratio} | {beam_size} | {k_number} | {seed_number} | {' | '.join(metrics)} | {max_tokens} | {num_samples} | {skip_samples} |\n"
                )

            else:
                continue

        if metrics:
            
            if evaluator == "chair":
                markdown_table += f"| {backbone} | {decoder} | {detector} | {box} | {expand_ratio} | {beam_size} | {k_number} | {seed_number} | {' | '.join(metrics)} | {max_tokens} | {num_samples} |\n"
                writer.writerow(
                    {
                        "Backbone": backbone,
                        "Decoder": decoder,
                        "Detector": detector,
                        "Box": box,
                        "Ratio": expand_ratio,
                        "Beam": beam_size,
                        "K": k_number,
                        "Seed": seed_number,
                        "SPICE": metrics[0],
                        "METEOR": metrics[1],
                        "CIDEr": metrics[2],
                        "CHAIRs": metrics[3],
                        "CHAIRi": metrics[4],
                        "BLEU": metrics[5],
                        "Log CIDEr": metrics[6],
                        "Max Tokens": max_tokens,
                        "Num of Samples": num_samples,
                    }
                )

            elif evaluator == "pope":
                markdown_table += f"| {backbone} | {decoder} | {detector} | {box} | {expand_ratio} | {beam_size} | {k_number} | {seed_number} | {' | '.join(metrics)} | {max_tokens} | {num_samples} | {skip_samples} |\n"
                writer.writerow(
                    {
                        "Backbone": backbone,
                        "Decoder": decoder,
                        "Detector": detector,
                        "Box": box,
                        "Ratio": expand_ratio,
                        "Beam": beam_size,
                        "K": k_number,
                        "Seed": seed_number,
                        "Accuracy": metrics[0],
                        "Precision": metrics[1],
                        "Recall": metrics[2],
                        "F1 score": metrics[3],
                        "Yes ratio": metrics[4],
                        "Max Tokens": max_tokens,
                        "Num of Samples": num_samples,
                        "Skip Samples": skip_samples,
                    }
                )


# Save the markdown table to a file
if evaluator == "chair":
    markdown_file_path = "eval/eval_chair_results.md"
elif evaluator == "pope":
    markdown_file_path = "eval/eval_pope_results.md"
with open(markdown_file_path, "w") as md_file:
    md_file.write(markdown_table)

# The CSV file is already saved at this point, so we can output the paths to both files
print(f"Markdown file saved to: {markdown_file_path}")
print(f"CSV file saved to: {csv_file_path}")
