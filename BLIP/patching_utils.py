####################################################################################################
####################################################################################################

 ############# SOME PATCHING CODE IS ADAPTED FROM: https://github.com/kmeng01/rome.git #############

####################################################################################################
####################################################################################################


import contextlib
from torchvision.transforms.functional import to_pil_image
import copy
import inspect
from collections import OrderedDict
import torch
from transformers import BlipForQuestionAnswering,BlipProcessor
import os, re, json
import torch, numpy
from collections import defaultdict
import numpy as np
import torch
from sklearn.decomposition import PCA
import matplotlib
from matplotlib import pyplot as plt
import itertools
import nltk
import random
# from sentence_transformers import SentenceTransformer
from tqdm.notebook import tqdm
#import stanza
#import datasets
import scipy
import csv
import pandas as pd
from PIL import Image
import pickle
from typing import Any, Optional, Tuple, Union,OrderedDict
import requests
import ast
from typing import Sequence, Tuple
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_triplet(triplet_str):
    try:
        # Attempt to parse it as a literal list; use the first element if it's a list
        parsed = ast.literal_eval(triplet_str)
        if isinstance(parsed, list) and len(parsed) > 0:
            return parsed[0].split(',')
        else:
            return triplet_str.split(',')
    except:
        # Fallback for non-list strings or if ast.literal_eval fails
        return triplet_str.split(',')


def get_incorrect_answers(df):
    incorrect_answers = []

    for index, row in df.iterrows():
        #@MICHAL ADDED THIS IF STATEMENT.
        # IF CORRECT_ANSWER is not CORRECT_ANSWER_MC
        # then, the incorrect_token=correct_answers
        if row['correct_answer'] != row['correct_answer_MC']:
            incorrect_answers.append(row['correct_answer'])
        else:
            # Determine the correct index based on negation flags
            incorrect_index = 0 if row['subj_neg'] else (1 if row['verb_neg'] else 2)

            # Process for positive triplet
            pos_triplet = parse_triplet(row['pos_triplet'])
            neg_triplet = parse_triplet(row['neg_triplet'])
            incorrect_answers.append(neg_triplet[incorrect_index])

    return incorrect_answers
 

class StopForward(Exception):
    """
    If the only output needed from running a network is the retained
    submodule then Trace(submodule, stop=True) will stop execution
    immediately after the retained submodule by raising the StopForward()
    exception.  When Trace is used as context manager, it catches that
    exception and can be used as follows:

    with Trace(net, layername, stop=True) as tr:
        net(inp) # Only runs the network up to layername
    print(tr.output)
    """

    pass


def recursive_copy(x, clone=None, detach=None, retain_grad=None):
    """
    Copies a reference to a tensor, or an object that contains tensors,
    optionally detaching and cloning the tensor(s).  If retain_grad is
    true, the original tensors are marked to have grads retained.
    """
    if not clone and not detach and not retain_grad:
        return x
    if isinstance(x, torch.Tensor):
        if retain_grad:
            if not x.requires_grad:
                x.requires_grad = True
            x.retain_grad()
        elif detach:
            x = x.detach()
        if clone:
            x = x.clone()
        return x
    # Only dicts, lists, and tuples (and subclasses) can be copied.
    if isinstance(x, dict):
        return type(x)({k: recursive_copy(v) for k, v in x.items()})
    elif isinstance(x, (list, tuple)):
        return type(x)([recursive_copy(v) for v in x])
    else:
        assert False, f"Unknown type {type(x)} cannot be broken into tensors."


def subsequence(
    sequential,
    first_layer=None,
    last_layer=None,
    after_layer=None,
    upto_layer=None,
    single_layer=None,
    share_weights=False,
):
    """
    Creates a subsequence of a pytorch Sequential model, copying over
    modules together with parameters for the subsequence.  Only
    modules from first_layer to last_layer (inclusive) are included,
    or modules between after_layer and upto_layer (exclusive).
    Handles descent into dotted layer names as long as all references
    are within nested Sequential models.

    If share_weights is True, then references the original modules
    and their parameters without copying them.  Otherwise, by default,
    makes a separate brand-new copy.
    """
    assert (single_layer is None) or (
        first_layer is last_layer is after_layer is upto_layer is None
    )
    if single_layer is not None:
        first_layer = single_layer
        last_layer = single_layer
    first, last, after, upto = [
        None if d is None else d.split(".")
        for d in [first_layer, last_layer, after_layer, upto_layer]
    ]
    return hierarchical_subsequence(
        sequential,
        first=first,
        last=last,
        after=after,
        upto=upto,
        share_weights=share_weights,
    )


def hierarchical_subsequence(
    sequential, first, last, after, upto, share_weights=False, depth=0
):
    """
    Recursive helper for subsequence() to support descent into dotted
    layer names.  In this helper, first, last, after, and upto are
    arrays of names resulting from splitting on dots.  Can only
    descend into nested Sequentials.
    """
    assert (last is None) or (upto is None)
    assert (first is None) or (after is None)
    if first is last is after is upto is None:
        return sequential if share_weights else copy.deepcopy(sequential)
    assert isinstance(sequential, torch.nn.Sequential), (
        ".".join((first or last or after or upto)[:depth] or "arg") + " not Sequential"
    )
    including_children = (first is None) and (after is None)
    included_children = OrderedDict()
    # A = current level short name of A.
    # AN = full name for recursive descent if not innermost.
    (F, FN), (L, LN), (A, AN), (U, UN) = [
        (d[depth], (None if len(d) == depth + 1 else d))
        if d is not None
        else (None, None)
        for d in [first, last, after, upto]
    ]
    for name, layer in sequential._modules.items():
        if name == F:
            first = None
            including_children = True
        if name == A and AN is not None:  # just like F if not a leaf.
            after = None
            including_children = True
        if name == U and UN is None:
            upto = None
            including_children = False
        if including_children:
            # AR = full name for recursive descent if name matches.
            FR, LR, AR, UR = [
                n if n is None or n[depth] == name else None for n in [FN, LN, AN, UN]
            ]
            chosen = hierarchical_subsequence(
                layer,
                first=FR,
                last=LR,
                after=AR,
                upto=UR,
                share_weights=share_weights,
                depth=depth + 1,
            )
            if chosen is not None:
                included_children[name] = chosen
        if name == L:
            last = None
            including_children = False
        if name == U and UN is not None:  # just like L if not a leaf.
            upto = None
            including_children = False
        if name == A and AN is None:
            after = None
            including_children = True
    for name in [first, last, after, upto]:
        if name is not None:
            raise ValueError("Layer %s not found" % ".".join(name))
    # Omit empty subsequences except at the outermost level,
    # where we should not return None.
    if not len(included_children) and depth > 0:
        return None
    result = torch.nn.Sequential(included_children)
    result.training = sequential.training
    return result


def set_requires_grad(requires_grad, *models):
    """
    Sets requires_grad true or false for all parameters within the
    models passed.
    """
    for model in models:
        if isinstance(model, torch.nn.Module):
            for param in model.parameters():
                param.requires_grad = requires_grad
        elif isinstance(model, (torch.nn.Parameter, torch.Tensor)):
            model.requires_grad = requires_grad
        else:
            assert False, "unknown type %r" % type(model)


def get_module(model, name):
    
    #print("GET MODULE NAME: ", name)
    
    """
    Finds the named module within the given model.
    """
    for n, m in model.named_modules():
        if n == name:
            return m
    raise LookupError(name)


def get_parameter(model, name):
    
    print("GET PARAMETER NAME: ", name)
    
    """
    Finds the named parameter within the given model.
    """
    for n, p in model.named_parameters():
        if n == name:
            return p
    raise LookupError(name)


def replace_module(model, name, new_module):
    """
    Replaces the named module within the given model.
    """
    if "." in name:
        parent_name, attr_name = name.rsplit(".", 1)
        model = get_module(model, parent_name)
    
    #print("ORIGINAL MODULE", getattr(model, attr_name))
    
    setattr(model, attr_name, new_module)
    
    #print("NEW MODULE", getattr(model, attr_name))
    

def invoke_with_optional_args(fn, *args, **kwargs):
    """
    Invokes a function with only the arguments that it
    is written to accept, giving priority to arguments
    that match by-name, using the following rules.
    (1) arguments with matching names are passed by name.
    (2) remaining non-name-matched args are passed by order.
    
    
    # WILLIAM Lets look into this for crossattention and see if 3 is triggered. 
    # I think this is working fine, but not 1000% confident. 
    
    (3) extra caller arguments that the function cannot
        accept are not passed.
    (4) extra required function arguments that the caller
        cannot provide cause a TypeError to be raised.
    Ordinary python calling conventions are helpful for
    supporting a function that might be revised to accept
    extra arguments in a newer version, without requiring the
    caller to pass those new arguments.  This function helps
    support function callers that might be revised to supply
    extra arguments, without requiring the callee to accept
    those new arguments.
    """
    argspec = inspect.getfullargspec(fn)
    pass_args = []
    used_kw = set()
    unmatched_pos = []
    used_pos = 0
    defaulted_pos = len(argspec.args) - (
        0 if not argspec.defaults else len(argspec.defaults)
    )
    # Pass positional args that match name first, then by position.
    for i, n in enumerate(argspec.args):
        if n in kwargs:
            pass_args.append(kwargs[n])
            used_kw.add(n)
        elif used_pos < len(args):
            pass_args.append(args[used_pos])
            used_pos += 1
        else:
            unmatched_pos.append(len(pass_args))
            pass_args.append(
                None if i < defaulted_pos else argspec.defaults[i - defaulted_pos]
            )
    # Fill unmatched positional args with unmatched keyword args in order.
    if len(unmatched_pos):
        for k, v in kwargs.items():
            if k in used_kw or k in argspec.kwonlyargs:
                continue
            pass_args[unmatched_pos[0]] = v
            used_kw.add(k)
            unmatched_pos = unmatched_pos[1:]
            if len(unmatched_pos) == 0:
                break
        else:
            if unmatched_pos[0] < defaulted_pos:
                unpassed = ", ".join(
                    argspec.args[u] for u in unmatched_pos if u < defaulted_pos
                )
                raise TypeError(f"{fn.__name__}() cannot be passed {unpassed}.")
    # Pass remaining kw args if they can be accepted.
    pass_kw = {
        k: v
        for k, v in kwargs.items()
        if k not in used_kw and (k in argspec.kwonlyargs or argspec.varargs is not None)
    }
    # Pass remaining positional args if they can be accepted.
    if argspec.varargs is not None:
        pass_args += list(args[used_pos:])
    return fn(*pass_args, **pass_kw)

class Trace(contextlib.AbstractContextManager):
    """
    To retain the output of the named layer during the computation of
    the given network:

        with Trace(net, 'layer.name') as ret:
            _ = net(inp)
            representation = ret.output

    A layer module can be passed directly without a layer name, and
    its output will be retained.  By default, a direct reference to
    the output object is returned, but options can control this:

        clone=True  - retains a copy of the output, which can be
            useful if you want to see the output before it might
            be modified by the network in-place later.
        detach=True - retains a detached reference or copy.  (By
            default the value would be left attached to the graph.)
        retain_grad=True - request gradient to be retained on the
            output.  After backward(), ret.output.grad is populated.

        retain_input=True - also retains the input.
        retain_output=False - can disable retaining the output.
        edit_output=fn - calls the function to modify the output
            of the layer before passing it the rest of the model.
            fn can optionally accept (output, layer) arguments
            for the original output and the layer name.
        stop=True - throws a StopForward exception after the layer
            is run, which allows running just a portion of a model.
    """

    def __init__(
        self,
        module,
        layer=None,
        retain_output=True,
        retain_input=False,
        clone=False,
        detach=False,
        retain_grad=False,
        edit_output=None,
        stop=False,
    ):
        """
        Method to replace a forward method with a closure that
        intercepts the call, and tracks the hook so that it can be reverted.
        """
        retainer = self
        self.layer = layer
        if layer is not None:
            module = get_module(module, layer)
        
        def retain_hook(m, inputs, output):
            #print(f"Hooked layer: {self.layer}")
            if retain_input:
                retainer.input = recursive_copy(
                    inputs[0] if len(inputs) == 1 else inputs,
                    clone=clone,
                    detach=detach,
                    retain_grad=False,
                )  # retain_grad applies to output only.
            
                
            if edit_output:

                
                output = invoke_with_optional_args(
                    edit_output, output=output, layer=self.layer
                )
                
                
                
            if retain_output:
                retainer.output = recursive_copy(
                    output, clone=clone, detach=detach, retain_grad=retain_grad
                )
                
                
                if retain_grad:
                    output = recursive_copy(retainer.output, clone=True, detach=False)
                    
            if stop:
                raise StopForward()
            return output

        self.registered_hook = module.register_forward_hook(retain_hook)
        self.stop = stop

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()
        if self.stop and issubclass(type, StopForward):
            return True

    def close(self):
        self.registered_hook.remove()

        
class TraceDict(OrderedDict, contextlib.AbstractContextManager):
    """
    To retain the output of multiple named layers during the computation
    of the given network:

        with TraceDict(net, ['layer1.name1', 'layer2.name2']) as ret:
            _ = net(inp)
            representation = ret['layer1.name1'].output

    If edit_output is provided, it should be a function that takes
    two arguments: output, and the layer name; and then it returns the
    modified output.

    Other arguments are the same as Trace.  If stop is True, then the
    execution of the network will be stopped after the last layer
    listed (even if it would not have been the last to be executed).
    """

    def __init__(
        self,
        module,
        layers=None,
        retain_output=True,
        retain_input=False,
        clone=False,
        detach=False,
        retain_grad=False,
        edit_output=None,
        stop=False,
    ):
        self.stop = stop

        def flag_last_unseen(it):
            try:
                it = iter(it)
                prev = next(it)
                seen = set([prev])
            except StopIteration:
                return
            for item in it:
                if item not in seen:
                    yield False, prev
                    seen.add(item)
                    prev = item
            yield True, prev
 
        for is_last, layer in flag_last_unseen(layers):
            self[layer] = Trace(
                module=module,
                layer=layer,
                retain_output=retain_output,
                retain_input=retain_input,
                clone=clone,
                detach=detach,
                retain_grad=retain_grad,
                edit_output=edit_output,
                stop=stop and is_last,
            )

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()
        if self.stop and issubclass(type, StopForward):
            return True

    def close(self):
        for layer, trace in reversed(self.items()):
            trace.close()
            
            

def layername(model, num, block_name, kind=None):

    if not hasattr(model, block_name):
        raise ValueError(f"The model does not have an attribute '{block_name}'")


    base_paths = {
        "vision_model": "vision_model",
        "text_encoder": "text_encoder.encoder.layer",
        "text_decoder": "text_decoder.bert.encoder.layer"
    }
    

    special_kinds = {
        "embed": {
            "vision_model": "vision_model.embeddings",
            "text_encoder": "text_encoder.embeddings",
            "text_decoder": "text_decoder.bert.embeddings"
        },
        "cls": {
            "text_decoder": "text_decoder.cls"
        },
        "crossattention_layernorm": {
            "text_encoder": f"text_encoder.encoder.layer.{num}.crossattention.output.LayerNorm"
        },
        "crossattention_block": {
            "text_encoder": f"text_encoder.encoder.layer.{num}.crossattention.self"
        },
        "attention_block": {
            "text_encoder": f"text_encoder.encoder.layer.{num}.attention.self"
        }
    }
    
    
    if kind in special_kinds and block_name in special_kinds[kind]:
        return special_kinds[kind][block_name]
    

    base_path = base_paths.get(block_name)
    if base_path is None:
        raise ValueError(f"Unknown transformer architecture: '{block_name}'")

    
    layer_path = f"{base_path}.{num}"
    if kind not in ["embed", "cls", None]:  # Already handled by special_kinds
        layer_path += f".{kind}"
    
    return layer_path


def predict_from_input(model, pixel_values,input_ids,attention_mask=None):
    output=model.forward(pixel_values=pixel_values.to(device),input_ids=input_ids.to(device))
    #attention_mask=processor(text=text_input, add_special_tokens=False).attention_mask

    return output

def result_gen(output, return_p=False):
    out=output['decoder_logits']
    probs = torch.softmax(out[:, -1], dim=1)
    p, preds = torch.max(probs, dim=1)
    return preds, p

def result_gen_min(output, return_p=False):
    out=output['decoder_logits']
    probs = torch.softmax(out[:, -1], dim=1)
    p, preds = torch.min(probs, dim=1)
    return preds, p

def decoding(pred_prob):
    preds=pred_prob[0]
    ans=[]
    for i in range(preds.size(0)):
        single_pred=preds[i]
        decoded_answer=processor.decode(single_pred, skip_special_tokens=True)
        ans.append(decoded_answer)
    return ans[0]

def trace_with_patch(
    model,processor,constant_input,  # The model
    clean_input, corrupt_input, answer_tokens, mode, # A set of inputs
    states_to_patch,  # A list of (token index, layername) triples to restore # Answer probabilities to collect  # List of traced outputs to return
    attn_head=None, # index of the attention HEAD we want to patch. None if we patch all heads or don't patch attention. 
    ):
    """
    Runs a single causal trace.  Given a model and a batch input where
    the batch size is at least two, runs the batch in inference, corrupting
    a the set of runs [1...n] while also restoring a set of hidden states to
    the values from an uncorrupted run [0] in the batch.

    The convention used by this function is that the zeroth element of the
    batch is the uncorrupted run, and the subsequent elements of the batch
    are the corrupted and to-be-patched runs.

    The constant input is the modality which is uncorrupted - 
        if mode =="image" : text is the constant input
        elif mode=="text" : image is the constant input

    Then when running, a specified set of hidden states will be uncorrupted
    by restoring their values to the same vector that they had in the
    zeroth uncorrupted run.  This set of hidden states is listed in
    states_to_patch, by listing [(token_index, layername), ...] pairs.
    To trace the effect of just a single state, this can be just a single
    token/layer pair.  To trace the effect of restoring a set of states,
    any number of token indices and layers can be listed.

    answer_tokens depicts the tokenized ids of the clean and corrupted answers whose probabilities
    needs to be obtained

    attn_head passes the attn_head(cross) id which will be hooked from the model
    """


    patch_spec = defaultdict(list)
    for t, l in states_to_patch:
        patch_spec[l].append(t)
    
    if mode == 'image':
        processed_inputs = processor(images=[clean_input, corrupt_input, corrupt_input], text=[constant_input]*3, return_tensors='pt', padding=True)
    elif mode == 'text':
        processed_inputs = processor(images=[constant_input]*3, text=[clean_input, corrupt_input, corrupt_input], return_tensors='pt', padding=True)
    else:
        print("MODE IS EQUAL TO: ", mode) 
        raise ValueError("Invalid mode specified. Use 'image' or 'text'.")

    pixel_val_tensor = processed_inputs.pixel_values.to(device)
    input_ids_tensor = processed_inputs.input_ids.to(device)
    
    input_ids_tensor=input_ids_tensor.to(device)
    pixel_val_tensor=pixel_val_tensor.to(device)

    def untuple(x):
        return x[0] if isinstance(x, tuple) else x
    
    def transpose_for_scores(x):
        # This function is taken directly from BlipTextSelfAttention
        # these are hard coded for blip. 
        num_attention_heads = 12
        attention_head_size = 64
        # breaking into attention heads. 
        new_x_shape = x.size()[:-1] + (num_attention_heads, attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def inverse_transpose_for_scores(x):
        # This operation is the inverse of transpose for scores. 
        #og_shape = [batch, num_tokens, all_head_size]
        x = x.permute(0,2,1,3)
        og_shape = x.shape[:2] + (x.shape[2]*x.shape[3],)
        return x.view(og_shape)
        
    def patch_rep(x, layer, attn_head=attn_head): 
        if layer not in patch_spec:
            return x

        h = untuple(x)  
        if h.size(0) == 1:
            return x  

        patched = False    
        for t in patch_spec[layer]:
            # The untuple only selects the first hidden state for attention.
            # attn_ head not None bc if attn_head = 0 (index 0) this is not triggered 
            if attn_head is not None:
                # break output into attention heads:
                h = transpose_for_scores(h)
                # Patch attn_head at token t
                h[2,attn_head,t,:] = h[0,attn_head,t,:].clone()
                # reshape back to the original embedding shape.
                h = inverse_transpose_for_scores(h)
            else:
                # h = [clean, corrupt, corrupt]
                h[2, t] = h[0, t].clone()  #Use clone() to ensure a distinct tensor is assigned.
                # the h being returned is h=[clean, corrupt, patched]
            patched = True

        if patched:
            return (h,) + x[1:] if isinstance(x, tuple) else h 
        else:
            return x

    with torch.no_grad(),TraceDict(
        model,
        list(patch_spec.keys()),
        edit_output=patch_rep,
    ) as td:
        
        
        outputs_exp = model.forward(pixel_values=pixel_val_tensor,input_ids=input_ids_tensor, output_attentions=True)
    logits = outputs_exp['decoder_logits'][:, -1, :]
    
    # Get correct and incorrect answer tokens.  
    correct_token = answer_tokens[0]
    incorrect_token = answer_tokens[1]

    prob_clean_correct = torch.softmax(logits[0], dim=0)[correct_token]
    prob_corrupt_correct = torch.softmax(logits[1], dim=0)[correct_token]
    # Prob_patched_correct is what we originally return.
    prob_patched_correct = torch.softmax(logits[2], dim=0)[correct_token]
    
    # Probability Difference := P_patched(correct_token) - P_corrupt(correct_token)
    p_diff = prob_patched_correct - prob_corrupt_correct
    
    # Logit Difference: 
    logits_clean = logits[0].squeeze()
    logits_corrupt = logits[1].squeeze()
    logits_patched = logits[2].squeeze()
    
    LD_clean = logits_clean[correct_token] - logits_clean[incorrect_token] 
    LD_corrupt = logits_corrupt[correct_token] - logits_corrupt[incorrect_token]
    LD_patched = logits_patched[correct_token] - logits_patched[incorrect_token]
    
    # This is from Best Practices 
    logit_diff = LD_patched - LD_corrupt
    patching_effect = logit_diff/(LD_clean - LD_corrupt)

    # p_diff should probably be very very close to prob_patched_correct bc prob__corrupt_correct should be small
    # logit_diff is the 'raw' logit diff
    # patching_effect is a pseudo-normalized version of logit diff (from best practices)
    #[prob_clean_correct, prob_patched_correct, p_diff, logit_diff]

    return torch.tensor([prob_patched_correct, p_diff, logit_diff, patching_effect])




def trace_important_states(model,processor, constant_input, clean_input, corrupt_input, answer_tokens, mode, start,num_layers,block_name,num=None,kind=None
    , attn_head=None):
    
    outputs = []

    table = []

    if mode == 'image':
        input_ids=processor(text=constant_input,return_tensors='pt').input_ids

    elif mode == 'text':
        input_ids=processor(text=clean_input,return_tensors='pt').input_ids #using question1 bc it doesn't matter  
    else:
        raise ValueError("Invalid mode specified. Use 'image', or 'text'.")


    if num==None:
        num=input_ids.size(1)
    for tnum in range(num):
        row = []
        for layer in range(start,num_layers):

            scores = trace_with_patch(
                model,processor, constant_input, clean_input, corrupt_input, answer_tokens, mode,
                [(tnum, layername(model,layer,block_name,kind))], attn_head,
            )
            row.append(scores)
        table.append(torch.stack(row))

    return torch.stack(table) 

def calculate_hidden_flow(model, processor, constant_input, clean_input, corrupt_input, answer_tokens, mode, start, num_layers, block_name, kind=None, attn_head=None):
    '''
    low_score gets the unpatched corrupt scores
    '''
    low_score = trace_with_patch(model, processor, constant_input, clean_input, corrupt_input, answer_tokens=answer_tokens, mode=mode, states_to_patch=[], attn_head=attn_head)
    '''
    This performs path patching for the blocks inside the transformer namely text_encoder, text_decoder and vision_model
    '''
    if block_name in ['text_encoder', 'text_decoder', 'vision_model']:
        scores = trace_important_states(
            model, processor, constant_input, clean_input, corrupt_input, answer_tokens, mode, start, num_layers, block_name, kind=kind, attn_head=attn_head
        )
    else:
        raise ValueError(f"Unknown block_name: {block_name}")
    
    if mode == 'image':
        text=constant_input
    else:
        text=clean_input 
        
    return {
        "scores": scores,
        "low_score": low_score,
        "question": text,
        "correct_token": answer_tokens[0],
        "incorrect_token": answer_tokens[1],
        "block_name": block_name,
        "kind": kind,
        "attn_head": attn_head,
    }

def debug_hidden_flow(model, processor, df, block_name, kind,start, end, mode,  attn_head=None, image_tensors=None):
    temp_list = []
    outputs = []
    
    for idx, row in df.iterrows():
        if mode == 'image':
            # this is for SVO probes where we store image tensors locally instead of file_paths
            if image_tensors:
                image_tensor1 = image_tensors.get(row["clean_image_path"])
                clean_input = to_pil_image(image_tensor1)
                image_tensor2 = image_tensors.get(row["corrupt_image_path"])
                corrupt_input = to_pil_image(image_tensor2)
            
            else:
                image_path1 = row["clean_image_path"]
                clean_input = Image.open(image_path1)
                image_path2 = row["corrupt_image_path"]
                corrupt_input = Image.open(image_path2)
                constant_input = row['clean_prompt']
    
        elif mode == 'text':
            if image_tensors:
                image_tensor = image_tensors.get(row["clean_image_path"])
                constant_input = to_pil_image(image_tensor)
            else:
                image_path = row["clean_image_path"]
                constant_input = Image.open(image_path)

            clean_input = row['clean_prompt'] 
            corrupt_input = row['corrupt_prompt']
        else:
            raise ValueError("Invalid mode specified.")

        answer_tokens = processor(text=[row["correct_answer"], row["incorrect_answer"]]).input_ids
        
        # We take index 1 because the processor returns [101, token_idx, 102], we we just take the token. 
        answer_tokens = [t[1] for t in answer_tokens]
         
        temp = calculate_hidden_flow(model, processor, constant_input, 
                                     clean_input, corrupt_input, answer_tokens=answer_tokens, start=start, num_layers=end, block_name=block_name, kind=kind, mode=mode, attn_head=attn_head)
        temp_list.append(temp)

        
    print("FINISHED PATCHING FOR KIND={}".format(str(kind)))
    print("FINISHED PATCHING FOR ATTN_HEAD={}".format(str(attn_head)))

    return temp_list