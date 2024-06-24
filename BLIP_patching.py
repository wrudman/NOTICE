import pandas as pd
import argparse
import numpy as np
import pickle
from transformers import BlipProcessor
from patching_utils import *
from models_patching import *

    
def main(task, samples, block_name, kind, mode, attn_head):
    
    model = ModifiedBlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
    model.to(device)
    processor = BlipProcessor.from_pretrained('Salesforce/blip-vqa-base')
    
    df_correct = pd.read_csv(f'BLIP_final_{task}.csv')
    
    if task == "svo_probes":
        filename = "image_tensors_500.pkl" if samples == "mini" else "image_tensors_final.pkl"

        with open(filename, 'rb') as f:
            image_tensors = pickle.load(f)

        image_id_set = set(dict(image_tensors).keys())
        image_tensors = dict(image_tensors)
        df_correct = df_correct[(df_correct.pos_url.isin(image_id_set)) & (df_correct.neg_url.isin(image_id_set))]
    else:
        image_tensors = None

    df_correct = df_correct.head(10) if samples == "mini" else df_correct
    df_correct["incorrect_answer"] = get_incorrect_answers(df_correct)

    temp_list = debug_hidden_flow(model, processor, df_correct, block_name=block_name, kind=kind, start=0, end=12, mode=mode, attn_head=attn_head, knockout=knockout, image_tensors=image_tensors)
    
    if attn_head is not None:  
        pickle_file_path = f'BLIP_temp_list_{task}_{mode}_corruption_{block_name}_{kind}_head_{attn_head}_{len(df_correct)}.pkl'
        print(pickle_file_path)
    else:
        pickle_file_path = f'BLIP_temp_list_{task}_{mode}_corruption_{block_name}_{kind}_{len(df_correct)}.pkl'
    with open(pickle_file_path, 'wb') as file:
        pickle.dump(temp_list, file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate heatmaps for BLIP model analysis.")
    parser.add_argument("--task", type=str, required=True, help="Task Options: svo_probes, facial_expressions, mit_states.")
    parser.add_argument("--samples", type=str, choices=["mini", "full"], required=True, help="Dataset size: 'mini' for a smaller subset, 'full' for the entire dataset.")
    parser.add_argument("--block_name", type=str, required=True, help="Name of the model block to analyze. Text encoder is what we use.")
    parser.add_argument("--kind", type=str, required=True, help="Type of analysis to perform on the block. Options: output, attention_block, crossattention_block.")
    parser.add_argument("--mode", type=str, required=True, help="text, image or knockout. Text means corrupting the text and image means corrupting the images.")
    # If we want we can modify this to patch a set of attention heads 
    parser.add_argument("--attn_head", type=int, default=None, help="Index of the attention you want to patch at each layer.")
    args = parser.parse_args()

    main(args.task, args.samples, args.block_name, args.kind, args.mode, args.attn_head)
