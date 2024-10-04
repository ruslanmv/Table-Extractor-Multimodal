import gradio as gr
import os
import torch
from transformers import AutoProcessor, MllamaForConditionalGeneration, TextStreamer
from PIL import Image
import csv

# # Table Extractor Multimodal
# The purpose of this program is extract the tables contained in a image or pdf.
# 
# The pipeline is the follows:
# - First the program convert the pdf to image
# - The image is analized by the llm multimodal
# - It is extracted all the tables in format csv
# - It is preprocessed the output of the llm into csv and saved all individual tables.
# 
#Package installation
#!pip install git+https://github.com/huggingface/transformers.git
#!pip install torch, accelerate, bitsandbyte, sentencepiece, pillow
#!pip install spaces


# Check if we're running in a Hugging Face Space and if SPACES_ZERO_GPU is enabled
IS_SPACES_ZERO = os.environ.get("SPACES_ZERO_GPU", "0") == "1"
IS_SPACE = os.environ.get("SPACE_ID", None) is not None
IS_GDRVIE = True

# Determine the device (GPU if available, else CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
LOW_MEMORY = os.getenv("LOW_MEMORY", "0") == "1"
print(f"Using device: {device}")
print(f"Low memory mode: {LOW_MEMORY}")

# Get Hugging Face token from environment variables
HF_TOKEN = os.environ.get('HF_TOKEN')

# Define the model name
model_name = "Llama-3.2-11B-Vision-Instruct"
if IS_GDRVIE:
    # Define the path to the model directory in your Google Drive
    model_path = "/content/drive/MyDrive/models/" + model_name
    model = MllamaForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_path)
else:
    model_name = "ruslanmv/" + model_name
    model = MllamaForConditionalGeneration.from_pretrained(
        model_name,
        use_auth_token=HF_TOKEN,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_name, use_auth_token=HF_TOKEN)

# Tie the model weights to ensure the model is properly loaded
if hasattr(model, "tie_weights"):
    model.tie_weights()

# Stream LLM response generator
def stream_response(inputs):
    streamer = TextStreamer(tokenizer=processor.tokenizer)
    for token in model.generate(**inputs, max_new_tokens=2000, do_sample=True, streamer=streamer):
        yield processor.decode(token, skip_special_tokens=True)

def predict(message, image):
    # Prepare the input messages
    messages = [
        {"role": "user", "content": [
            {"type": "image"},  # Specify that an image is provided
            {"type": "text", "text": message}  # Add the user-provided text input
        ]}
    ]

    # Create the input text using the processor's chat template
    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)

    # Process the inputs and move to the appropriate device
    inputs = processor(image, input_text, return_tensors="pt").to(device)

    # Return a streaming generator of responses
    return stream_response(inputs)    


files_list = []

def extract_and_save_tables(full_response):
    """Extracts CSV tables from the full_response string and saves them as separate files."""
    current_table_name = None
    current_table_rows = []

    for line in full_response.splitlines():
        if line.startswith("Table "):
            if current_table_name:
                # Save the previous table
                save_table_to_csv(current_table_name, current_table_rows)
                files_list.append(current_table_name)  # Add file name to the list
            
            # Extract the table number to create the filename
            current_table_name = "table_" + line.split("Table ")[1].replace(":", "").strip() + ".csv"
            current_table_rows = []
        elif current_table_name:
            # If it's not an empty line, add it to the current table rows
            if line.strip():
                current_table_rows.append(line)

    # Save the last table
    if current_table_name:
        save_table_to_csv(current_table_name, current_table_rows)
        files_list.append(current_table_name)  # Add file name to the list

def save_table_to_csv(table_name, table_rows):
    """Saves a table to a CSV file."""
    try:
        with open(table_name, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write each row to the CSV file
            for row in table_rows:
                writer.writerow(row.split(","))
        print(f"Table saved as: {table_name}")
    except Exception as e:
        print(f"Error saving table {table_name}: {e}")


def display_first_5_rows(filename):
  """Displays the first 5 rows of a CSV file."""
  try:
    with open(filename, 'r', encoding='utf-8') as csvfile:
      reader = csv.reader(csvfile)
      rows = list(reader)
      for i in range(min(5, len(rows))):
        print(rows[i])
      print("-" * 20)  # Separator between files
  except FileNotFoundError:
    print(f"File not found: {filename}")
  except Exception as e:
    print(f"Error reading file {filename}: {e}")

# ... (Your existing code for model loading, predict function, etc.) ...

def process_image(image):

    example='''Table 1:
    value1,value2,value3

    Table 2:
    value1,value2,value3
    '''
    message = "Please generate the csv file of all tables. You can include some rows with empty values. You can separate the tables by table_n.csv: then the table in csv. Print only the csv files. For example "+example + " Keep the name of original headers"



    full_response = ""
    
    # Update the process_image function to display the streaming output
    with gr.Blocks() as inner_block:
        streaming_output = gr.Textbox(label="LLM Streaming Output")
        for response in predict(message, image):
            print(response, end="", flush=True)
            full_response += response
            streaming_output.update(value=full_response)  # Update the output in the Gradio app
            
    extract_and_save_tables(full_response)

    header_info = ""
    for filename in files_list:
        try:
            with open(filename, 'r', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile)
                rows = list(reader)
                if rows:
                    header_info += f"**{filename}:**\n"
                    header_info += ", ".join(rows[0]) + "\n\n"
        except FileNotFoundError:
            header_info += f"File not found: {filename}\n"
        except Exception as e:
            header_info += f"Error reading file {filename}: {e}\n"

    return header_info, files_list

def download_files(files_list):
    file_paths = [os.path.abspath(file) for file in files_list]
    return gr.File.update(value=file_paths, visible=True)

with gr.Blocks() as demo:
    with gr.Row():
        image_input = gr.Image(type="pil", label="Upload Image")
        with gr.Column():
            header_output = gr.Textbox(label="Headers of Extracted Tables")
            download_button = gr.File(label="Download CSV Files", visible=False)

    process_button = gr.Button("Extract Tables")
    process_button.click(fn=process_image, inputs=image_input, outputs=[header_output, download_button])
    download_button.change(fn=download_files, inputs=download_button, outputs=download_button)

demo.launch(debug=True)

