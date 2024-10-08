# Table Extractor Using LLaMA-3.2 Multimodal

## Introduction

This project demonstrates how to convert images containing tables into CSV files using a multimodal model, **LLaMA-3.2**. The guide explains step by step how to install necessary dependencies, run the provided code, and utilize **LLaMA-3.2** to extract tables from images and save them as CSV files. This approach works with images or PDF documents that are converted into images.

### What is Multimodal?

Multimodal models can process different types of inputs simultaneously, such as images, text, and audio. In this case, the multimodal model **LLaMA-3.2** processes both image inputs and text instructions to analyze image content and extract tables into CSV format. This approach combines visual understanding and language comprehension, enabling models to analyze tables from images.

---

## Full Python Code

### Setup and Installation

#### Step 1: Install Required Packages

To run the project, first install the required dependencies:

```bash
pip install git+https://github.com/huggingface/transformers.git
pip install torch accelerate bitsandbytes sentencepiece pillow gradio
```

These packages include the necessary tools for model loading, processing images, and building the web interface.

---

## Code Explanation

### 1. **Model Setup**

The first step is setting up the model and determining the computing device (GPU or CPU). The code loads the **LLaMA-3.2** model and processor for processing text and image inputs.

```python
import os
import torch
from transformers import AutoProcessor, MllamaForConditionalGeneration

# Determine the device (GPU if available, else CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Get Hugging Face token from environment variables
HF_TOKEN = os.environ.get('HF_TOKEN')

# Define the model name
model_name = "Llama-3.2-11B-Vision-Instruct"
model = MllamaForConditionalGeneration.from_pretrained(
    model_name,
    use_auth_token=HF_TOKEN,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_name, use_auth_token=HF_TOKEN)
```

### 2. **Processing Input and Predicting Tables**

Here, the model is configured to process both image and text inputs. It takes a user-provided image (e.g., a photo of a document with tables) and a text instruction for the model to extract the tables.

```python
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
```

### 3. **Streaming Response**

The following code streams the model's output, simulating the conversation between the user and the multimodal model. This function reads the tokens generated by the model and yields the complete response.

```python
from transformers import TextStreamer

def stream_response(inputs):
    streamer = TextStreamer(tokenizer=processor.tokenizer)
    for token in model.generate(**inputs, max_new_tokens=2000, do_sample=True, streamer=streamer):
        yield processor.decode(token, skip_special_tokens=True)
```

### 4. **Extracting Tables and Saving as CSV**

Once the tables are extracted from the model’s output, the next step is to parse and save them as CSV files. The `extract_and_save_tables` function processes the text output to extract rows and columns from each table and save them as separate CSV files.

```python
import csv

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
```

### 5. **Displaying Extracted Data**

You can display the first few rows of each extracted table using the `display_first_5_rows` function. This helps in verifying if the tables have been correctly extracted.

```python
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
```

### 6. **User Interface with Gradio**

The Gradio interface allows users to upload an image, process it using the model, and download the extracted tables as CSV files. The user uploads an image, clicks the "Extract Tables" button, and the CSV files are generated and displayed for download.

```python
import gradio as gr
from PIL import Image

def process_image(image):
    message = "Please generate the CSV file of all tables."
    full_response = ""
    
    # Update the process_image function to display the streaming output
    for response in predict(message, image):
        print(response, end="", flush=True)
        full_response += response

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
```

---

## Step-by-Step Guide

1. **Upload the Image**: Use the Gradio interface to upload an image that contains tables.
2. **Extract Tables**: Once the image is uploaded, click the 'Extract Tables' button to start the table extraction process.
3. **Review the Headers**: After the tables are extracted, the headers of the tables will be displayed in the Gradio interface.
4. **Download CSV Files**: Once the tables have been saved, you can download the CSV files by clicking the 'Download CSV Files' button.

---

### Example Output

For an input image containing two tables, the model will output CSV files like:

- `table_1.csv`
- `table_2.csv`

Each CSV file will contain the corresponding rows and columns extracted from the image, which can then be viewed or processed further.

---

## Conclusion

This project showcases how to use the LLaMA-3.2 multimodal model to extract structured data from images. By following this guide, you can easily convert images containing tables into CSV files for further analysis and processing.
