import argparse
import pandas as pd
import pickle
import torch
from transformers import BitsAndBytesConfig, AutoProcessor, BlipForConditionalGeneration, BlipForQuestionAnswering
from torchvision.transforms.functional import to_pil_image
from PIL import Image
import io
import os

def generate_prompts(df, model, processor, task, image_tensors=None):
    device = next(model.parameters()).device
    df['generated_text'] = ""

    for index, row in df.iterrows():
        if task == "svo_probes":
            image_tensor = image_tensors.get(row['MC_pos_url'])
            if image_tensor is None:
                print(f"Image tensor not found for URL {row['pos_url']}")
                continue
            image_pil = to_pil_image(image_tensor)
        else:
            image_path = row["clean_image_path"]
            image_pil = Image.open(image_path)

        prompt = row['clean_prompt']
        try:
            inputs = processor(images=image_pil, text=prompt, return_tensors="pt").to(device)
            outputs = model.generate(**inputs)
            detached_output = outputs.detach()
            generated_text = processor.decode(detached_output[0], skip_special_tokens=True)

        except Exception as e:
            print(f"Error generating output for 'clean_prompt' at row {index}: {e}")
            generated_text = "Error in generation"
        
        df.at[index, 'generated_text'] = generated_text.strip()

    return df

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
    processor = AutoProcessor.from_pretrained("Salesforce/blip-vqa-base")

    model.to(device)

    df = pd.read_csv(f"{args.task}_cleaned.csv")

    image_tensors = None
    if args.task == "svo_probes":
        filename = "image_tensors_final.pkl"
        with open(filename, 'rb') as f:
            image_tensors = pickle.load(f)
        image_id_set = set(dict(image_tensors).keys())
        image_tensors = dict(image_tensors)
        df = df[(df.pos_url.isin(image_id_set)) & (df.neg_url.isin(image_id_set))]
    
    df = df.dropna()
    df_updated = generate_prompts(df, model, processor, args.task, image_tensors)

    output_file = f"BLIP_results_{args.task}.csv"
    df_updated.to_csv(output_file, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, help="Task Options: svo_probes, facial_expressions, mit_states")
    args = parser.parse_args()
    main(args)
