import os
import re
import subprocess
import csv

# Set the directory where the chair.json files are located
# directory = './paper_result/32_tokens/minigpt4/'
directory = '/home/czr/HaLC/paper_result/server'

# Function to run the eval_hallucination command and parse the output
def run_eval(file_path):
    # Running the eval_hallucination command for the given file
    result = subprocess.run(['python', 'eval_hallucination.py', '--chair_input_path', file_path],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    # print(result.stdout)
    # input()
    # Using regex to extract the metrics from the command output
    # metrics = re.findall(r'SPICE\s*(\d+\.\d+)\s*METEOR\s*(\d+\.\d+)\s*CIDEr\s*(\d+\.\d+)\s*CHAIRs\s*(\d+\.\d+)\s*CHAIRi\s*(\d+\.\d+)', 
    #                      result.stdout)
    metrics = re.findall(r'\d+\.\d+', result.stdout.split('ground truth captions')[-1].split('CHAIR results')[0])
    # Returning the extracted metrics
    return metrics if metrics else None

def extract_info_from_filename(filename):
    # Updated regex pattern to match various decoder names
    match = re.search(r'minigpt4_([a-zA-Z0-9-]+)_beams_(\d+)_k_(\d+)_coco_expand_ratio_([\d.]+)_seed_(\d+)_max', filename)
    if match:
        return match.group(1), int(match.group(2)), float(match.group(3)), float(match.group(4)), int(match.group(5))
    else:
        return '-', -1, -1, -1, -1

# Initialize the markdown table with headers
markdown_table = "| Decoder | Ratio | Beam | K | Seed | SPICE | METEOR | CIDEr | CHAIRs | CHAIRi |\n"
markdown_table += "|---------|-----------|-----------|----------|------------|-------|--------|-------|--------|--------|\n"

# Prepare the CSV file
csv_file_path = 'eval/eval_results.csv'
csv_columns = ['Decoder', 'Ratio', 'Beam', 'K', 'Seed', 'SPICE', 'METEOR', 'CIDEr', 'CHAIRs', 'CHAIRi']

# Start writing to the CSV file
with open(csv_file_path, 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
    writer.writeheader()  # write the header


    file_names = os.listdir(directory)
    # Sort the file names by beam size, then by k number, then by seed number
    sorted_file_names = sorted(file_names, key=lambda name: extract_info_from_filename(name)[1:])


    # Loop through each file in the directory and process it
    for file_name in sorted_file_names:
        if file_name.endswith('_chair.json'):
            file_path = os.path.join(directory, file_name)
            print(file_path)
            # Extract information from filename
            decoder,  beam_size, k_number, expand_ratio, seed_number = extract_info_from_filename(file_name)
            
            metrics = run_eval(file_path)
            
            print(f"| {decoder} | {expand_ratio} | {beam_size} | {k_number} | {seed_number} | {' | '.join(metrics)} |\n")

            if metrics:
                markdown_table += f"| {decoder} | {expand_ratio} | {beam_size} | {k_number} | {seed_number} | {' | '.join(metrics)} |\n"

                writer.writerow({
                    'Decoder': decoder,
                    'Ratio' : expand_ratio,
                    'Beam': beam_size,
                    'K': k_number,
                    'Seed': seed_number,
                    'SPICE': metrics[0],
                    'METEOR': metrics[1],
                    'CIDEr': metrics[2],
                    'CHAIRs': metrics[3],
                    'CHAIRi': metrics[4]
                })


# Save the markdown table to a file
markdown_file_path = 'eval/eval_results.md'
with open(markdown_file_path, 'w') as md_file:
    md_file.write(markdown_table)

# The CSV file is already saved at this point, so we can output the paths to both files
print(f'Markdown file saved to: {markdown_file_path}')
print(f'CSV file saved to: {csv_file_path}')