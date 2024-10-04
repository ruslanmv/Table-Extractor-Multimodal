#Package installation
#!pip install git+https://github.com/huggingface/transformers.git
#!pip install torch, accelerate, bitsandbyte, sentencepiece, pillow
#!pip install spaces
import gradio as gr
import os
import torch
from transformers import AutoProcessor, MllamaForConditionalGeneration, TextStreamer
from PIL import Image
import csv
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

example = '''Table 1:
header1,header2,header3
value1,value2,value3

Table 2:
header1,header2,header3
value1,value2,value3
'''

prompt_message = """Please extract all tables from the image and generate CSV files.
Each table should be separated using the format table_n.csv, where n is the table number.
You must use CSV format with commas as the delimiter. Do not use markdown format. Ensure you use the original table headers and content from the image.
Only answer with the CSV content. Dont explain the tables.
An example of the formatting output is as follows:
""" + example


# Stream LLM response generator
def stream_response(inputs):
    streamer = TextStreamer(tokenizer=processor.tokenizer)
    for token in model.generate(**inputs, max_new_tokens=2000, do_sample=True, streamer=streamer):
        yield processor.decode(token, skip_special_tokens=True)



# Predict function for Gradio app
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
    full_response = ""
    for response in stream_response(inputs):
       # print(response, end="", flush=True)  # Print each part of the response as it's generated
        full_response += response 
    return extract_and_save_tables(full_response) 

# Extract tables and save them to CSV
files_list = []

def clean_full_response(full_response):
    """Cleans the full response by removing the prompt input before the tables."""
    # The part of the prompt input to remove
    message_to_remove = prompt_message
    # Remove the message and return only the tables
    return full_response.replace(message_to_remove, "").strip()

def extract_and_save_tables(full_response):
    """Extracts CSV tables from the cleaned_response string and saves them as separate files."""
    cleaned_response = clean_full_response(full_response)
    files_list = []  # Initialize the list of file names
    tables = cleaned_response.split("Table ")  # Split the response by table sections

    for i, table in enumerate(tables[1:], start=1):  # Start with index 1 for "Table 1"
        table_name = f"table_{i}.csv"  # File name for the current table
        rows = table.strip().splitlines()[1:]  # Remove "Table n:" line and split the table into rows
        rows = [row.replace('"', '').split(",") for row in rows if row.strip()]  # Clean and split by commas

        # Save the table as a CSV file
        with open(table_name, mode="w", newline='') as file:
            writer = csv.writer(file)
            writer.writerows(rows)
        
        files_list.append(table_name)  # Append the saved file to the list

    return files_list


# Gradio interface
def gradio_app():
    def process_image(image):
        message = prompt_message
        files = predict(message, image)
        return "Tables extracted and saved as CSV files.", files
    # Input components
    image_input = gr.Image(type="pil", label="Upload Image")

    #message_input = gr.Textbox(lines=2, placeholder="Enter your message", value=message)
    output_text = gr.Textbox(label="Extraction Status")
    file_output = gr.File(label="Download CSV files")

    # Gradio interface
    iface = gr.Interface(
        fn=process_image,
        inputs=[image_input],
        outputs=[output_text, file_output],
        title="Table Extractor and CSV Converter",
        description="Upload an image to extract tables and download CSV files.",
        allow_flagging="never"
    )

    iface.launch(debug=True)


