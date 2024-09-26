####################################################################################################
####################################################################################################

 ############# SOME PATCHING CODE IS ADAPTED FROM: https://github.com/kmeng01/rome.git #############

####################################################################################################
####################################################################################################

from torchvision.transforms.functional import to_pil_image
import copy
from collections import OrderedDict, defaultdict
import torch
import numpy as np
from PIL import Image
import pickle
from typing import Any, Optional, Tuple, Union, Sequence
import contextlib
import inspect
from collections import OrderedDict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def clean_and_combine_triplets(pos_triplet, neg_triplet):
    # Remove brackets and extra spaces if present
    clean_pos_triplet = pos_triplet.replace('[', '').replace(']', '').replace("'", "").strip()
    clean_neg_triplet = neg_triplet.replace('[', '').replace(']', '').replace("'", "").strip()
    
    # Split by commas and combine
    combined_triplets = clean_pos_triplet.split(',')[:3] + clean_neg_triplet.split(',')[:3] #adding [:3] bc sometimes there are two sets of triplets, like "['man,run,sea', 'person,run,sea']"
    
    # Create a set to remove duplicates and return the result
    unique_triplets = set(combined_triplets)
    return unique_triplets

def generate_toi_matches(row, processor, task):
    """
    Given a row in a dataframe and task type, this function extracts the index of the tokens of interest (TOI) 
    in the chatgpt prompt. These are the tokens we perform activation patching on.
    
    task == "facial_expressions", "svo_probes", "mit_states"
    """
    
    # Get the prompt sentence
    sentence = row['clean_prompt']
    
    # Form the triplets based on the task type
    if task == "facial_expressions":
        triplets = {"Is", "this", "person", "feeling", row['correct_answer'], row["incorrect_answer"].lower(), "?"}
    
    elif task == "svo_probes":
        triplets = clean_and_combine_triplets(row['pos_triplet'], row['neg_triplet'])
    
    elif task == "mit_states":
        triplets = {"Is", "this", row['noun'], row['correct_answer'], row['incorrect_answer'].lower(), '?'}

    # Tokenize input prompt
    inputs = processor.tokenizer(sentence, return_tensors="pt")
    tokens = processor.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    
    idx_to_token = {idx: token for idx, token in enumerate(tokens)}
    
    # Initialize matches with the position of '▁or'
    matches = {'▁or': tokens.index('▁or') if '▁or' in tokens else None}
    
    # Iterate over tokens and triplets to find exact matches
    for token in tokens:
        clean_token = token.replace('##', '')
        for triplet in list(triplets):
            if clean_token == '▁' + triplet:
                matches[triplet] = tokens.index(token)
                triplets.remove(triplet)  # Remove matched triplet to avoid re-matching

    # Add special tokens like '?' (used in all tasks)
    special_tokens = {'?': None}
    for key in special_tokens.keys():
        if key in tokens:
            special_tokens[key] = tokens.index(key)
    matches.update(special_tokens)

    # Similarity matching for remaining triplets
    for triplet in triplets:
        options = [(t, i) for i, t in enumerate(tokens) if i not in matches.values()]
        options = [(t, i) for t, i in options if t.startswith('▁')]
        
        if not options:
            break
        
        tokenized_triplet = processor.tokenizer.tokenize(triplet)
        matched = False
        
        if tokenized_triplet[0] == "▁":
            spaces = [(t, i) for i, t in enumerate(tokens) if i not in matches.values() and t == '▁']
            for space in spaces:
                construct_triplet = []
                token, idx = space
                construct_triplet.append(token)
                idx += 1
                while idx_to_token[idx].startswith("▁") == False and idx_to_token[idx] not in ["?", ":", "."]:
                    construct_triplet.append(idx_to_token[idx])
                    if construct_triplet == tokenized_triplet:
                        matches[triplet] = idx
                        matched = True
                        break
                    else:
                        idx += 1
                if matched:
                    break

        if not matched:
            option_tokens, _ = zip(*options)
            clean_options = [t.replace('##', '') for t in option_tokens]
            best_match = process.extractOne(triplet, clean_options, scorer=fuzz.WRatio)

            if best_match and best_match[1] > 55:
                best_match_token = best_match[0]
                best_match_index = next(i for i, t in enumerate(tokens) if t.replace('##', '') == best_match_token and i not in matches.values())

                # Move the best match index to the boundary
                while idx_to_token[best_match_index + 1].startswith("▁") == False and idx_to_token[best_match_index + 1] not in ["?", ":", "."]:
                    best_match_index += 1
                matches[triplet] = best_match_index

    # Sort matches based on index values
    matches = dict(sorted(matches.items(), key=lambda item: item[1]))

    return matches, list(matches.values())

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
    
    #print("GET PARAMETER NAME: ", name)
    
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
        "text_decoder": "text_decoder.bert.encoder.layer",
        "language_model":"language_model.model.layers"
    }
    

    special_kinds = {
        "embed": {
            "vision_model": "vision_model.embeddings",
            "text_encoder": "text_encoder.embeddings",
            "text_decoder": "text_decoder.bert.embeddings",
            "language_model":"language_model.model.embed_tokens"
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
            "text_encoder": f"text_encoder.encoder.layer.{num}.attention.self",
            "language_model":f"language_model.model.layers.{num}.self_attn"
        },
        "mlp_block":{
            "language_model":f"language_model.model.layers.{num}.mlp"
        }
    }
    
    
    if kind in special_kinds and block_name in special_kinds[kind]:
        return special_kinds[kind][block_name]
    

    base_path = base_paths.get(block_name)
    if base_path is None:
        raise ValueError(f"Unknown transformer architecture: '{block_name}'")

    
    layer_path = f"{base_path}.{num}"
    if kind not in ["embed", "cls", None, "attention_block", "mlp_block"]:  # Already handled by special_kinds
        # NOTE KIND WAS BEING RETURNED AS A TUPLE
        layer_path += f".{kind[0]}"
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
    model, processor, constant_input,  # The model
    clean_input, corrupt_input, svo_tokens, mode,  # A set of inputs
    states_to_patch,  # A list of (token index, layername) triples to restore
    attn_head=None,  # index of the attention HEAD we want to patch. None if we patch all heads or don't patch attention.
    knockout=None
):
    """
    Runs a single causal trace.  Given a model and a batch input where
    the batch size is at least two, runs the batch in inference, corrupting
    the set of runs [1...n] while also restoring a set of hidden states to
    the values from an uncorrupted run [0] in the batch.

    The convention used by this function is that the zeroth element of the
    batch is the uncorrupted run, and the subsequent elements of the batch
    are the corrupted runs.  The argument tokens_to_mix specifies an
    element to be corrupted by adding Gaussian noise to the embedding for
    the batch inputs other than the first element in the batch.  Alternately,
    subsequent runs could be corrupted by simply providing different
    input tokens via the passed input batch.

    Then when running, a specified set of hidden states will be uncorrupted
    by restoring their values to the same vector that they had in the
    zeroth uncorrupted run.  This set of hidden states is listed in
    states_to_patch, by listing [(token_index, layername), ...] pairs.
    To trace the effect of just a single state, this can be just a single
    token/layer pair.  To trace the effect of restoring a set of states,
    any number of token indices and layers can be listed.
    """
    
    patch_spec = defaultdict(list)
    clean_hidden={}
    for t, l in states_to_patch:
        patch_spec[l].append(t)
        
    # LETS SEE WHAT THIS IS. 
    #print(states_to_patch)
    
    if mode == 'image':
        processed_inputs_clean = processor(images=clean_input, text=constant_input, return_tensors='pt', padding=True)
        # we don't need to call the processor twice.
        processed_inputs_corr= processor(images=corrupt_input, text=constant_input, return_tensors='pt', padding=True)
        #processed_inputs_patch=processor(images=corrupt_input, text=constant_input, return_tensors='pt', padding=True)
    elif mode == 'text':
        processed_inputs_clean = processor(images=constant_input, text=clean_input, return_tensors='pt', padding=True)
        processed_inputs_corr = processor(images=constant_input, text=corrupt_input, return_tensors='pt', padding=True)
        #processed_inputs_patch=processor(images=constant_input, text=corrupt_input, return_tensors='pt', padding=True)
    else:
        raise ValueError("Invalid mode specified. Use 'image' or 'text'.")

    pixel_val_clean = processed_inputs_clean.pixel_values.to(device)
    pixel_val_corr= processed_inputs_corr.pixel_values.to(device)
    #pixel_val_patch= processed_inputs_patch.pixel_values.to(device)


    input_ids_clean = processed_inputs_clean.input_ids.to(device)
    input_ids_corr = processed_inputs_corr.input_ids.to(device)
    #input_ids_patch= processed_inputs_patch.input_ids.to(device)


    attention_mask_clean = processed_inputs_clean.attention_mask.to(device)
    attention_mask_corr = processed_inputs_corr.attention_mask.to(device)
    #attention_mask_patch= processed_inputs_patch.attention_mask.to(device)
    
    if attn_head is not None:
        outputs_exp_clean = model(input_ids=input_ids_clean,pixel_values=pixel_val_clean,attention_mask=attention_mask_clean,output_attentions=True)
        # Output of the W_v projection 
        clean_states = outputs_exp_clean.attentions
        # ATTENTIONS SHAPE: [1, 32, 601,601] -> LEN: 32
    
    else:
        outputs_exp_clean = model(input_ids=input_ids_clean,pixel_values=pixel_val_clean,attention_mask=attention_mask_clean,output_hidden_states=True)
        # Output of the W_v projection 
        clean_states = outputs_exp_clean.hidden_states[1:]
    
    #HIDDEN STATE SHAPE: [1, 575 + num_text_tokens, 4096] -> LEN: 33
    # I think we only need the logits not hidden states
    outputs_exp_corr = model(input_ids=input_ids_corr,pixel_values=pixel_val_corr,attention_mask=attention_mask_corr, output_hidden_states=False) 
    
    def untuple(x):
        return x[0] if isinstance(x, tuple) else x

    def transpose_for_scores_llava(x):
        # This reshaping comes directly from LlamaAttention 
        # batch_size.
        bsz =  x.shape[0]
        # number of tokens in batch
        q_len =  x.shape[1]
        
        num_heads = 32
        head_dim = 128   
        
        return x.view(bsz, q_len, num_heads, head_dim)#.transpose(1, 2)    
    
    def patch_rep_llava(x, layer, attn_head=attn_head): 
        if layer not in patch_spec:
            print(f'{layer} IS NOT IN PATCH_SPEC')
            return x

        h = untuple(x)  
        
        # for clean state, we need to have the correct layer and the correct token. 
        # how do we get this... 
        
        patched = False    
        for t in patch_spec[layer]:
            # this gives us the index of the layer. 
            # I think +1 since there are 33 hidden states and the first is the lexical embedding. 
            #print(layer)
            layer_idx = int(layer.split('.')[-2]) #+ 1 
            
            # The untuple only selects the first hidden state for attention.
            # attn_ head not None bc if attn_head = 0 (index 0) this is not triggered 
            if attn_head is not None:
                
                #print("PERFORMING ATTENION HEAD PATCHING")
                # break output into attention heads:
                h = transpose_for_scores_llava(h)

                #print("clean_states shape:", len(clean_states))
                
                h[0,t,attn_head,:] = clean_states[layer_idx][0,t, attn_head, :].clone()
                
                # reshape back to the original embedding shape.
                h = torch.reshape(h, (1,-1,4096))
                #print(h.shape)
            else:
                # clean_state is a tuple of the model hidden states 
                # layer idx gets the layer, and token_idx is the specified token. 
                #print('states shape: ', clean_states[layer_idx].shape)
                h[0, t] = clean_states[layer_idx][0,t] 
            patched = True

        if patched:
            return (h,) + x[1:] if isinstance(x, tuple) else h 
        else:
            return x
    
    with torch.no_grad(), TraceDict(
        model,
        list(patch_spec.keys()),
        edit_output=patch_rep_llava,
    ) as td_patch:
        outputs_exp_patch = model(input_ids=input_ids_corr, pixel_values=pixel_val_corr, attention_mask=attention_mask_corr)
    
    logits_clean=outputs_exp_clean['logits'][:,-1,:].squeeze()
    logits_patched=outputs_exp_patch['logits'][:,-1,:].squeeze()
    logits_corrupt=outputs_exp_corr['logits'][:,-1,:].squeeze()

    correct_token = svo_tokens[0]
    incorrect_token = svo_tokens[1]

    prob_clean_correct = torch.softmax(logits_clean, dim=0)[correct_token]
    prob_patched_correct = torch.softmax(logits_patched, dim=0)[correct_token]
    prob_corrupt_correct = torch.softmax(logits_corrupt, dim=0)[correct_token]

    p_diff = prob_patched_correct - prob_corrupt_correct

    LD_clean = logits_clean[correct_token] - logits_clean[incorrect_token]
    LD_corrupt = logits_corrupt[correct_token] - logits_corrupt[incorrect_token]
    LD_patched = logits_patched[correct_token] - logits_patched[incorrect_token]

    logit_diff = LD_patched - LD_corrupt
    patching_effect = logit_diff / (LD_clean - LD_corrupt)

    return torch.tensor([prob_patched_correct, p_diff, logit_diff, patching_effect])


def trace_important_states(
    model, processor, constant_input, clean_input, corrupt_input, svo_tokens, mode, start, num_layers, block_name, num=None, kind=None, attn_head=None, knockout=None, correct_idx=None):

    outputs = []
    table = []

    if mode == 'image':
        input_ids = processor(text=constant_input, return_tensors='pt').input_ids
    elif mode == 'text':
        input_ids = processor(text=clean_input, return_tensors='pt').input_ids
    else:
        raise ValueError("Invalid mode specified. Use 'image' or 'text'.")

    if knockout is not None:
        for layer in range(start, num_layers):
            scores= trace_with_patch(
                model, processor, constant_input, clean_input, corrupt_input, svo_tokens, mode,
                [(0, layername(model, layer, block_name, kind))], attn_head, knockout
            )
            table.append(scores)
    else:
        if num is None:
            num = input_ids.size(1)

        # If we are patching attn_heads, we just want to patch correct_token_idx
        if attn_head is not None:
            idx_to_patch = [correct_idx]
        else:
            # the first 8 tokens are formatting tokens
            # e.g. <s> USER: <image> \n
            idx_to_patch = range(input_ids.size(1) - 8)
        
        for tnum in idx_to_patch:
            row = []
            hidden_states_row = []
            for layer in range(start, num_layers):
                
                scores= trace_with_patch(
                    model, processor, constant_input, clean_input, corrupt_input, svo_tokens, mode,
                    # BIG Q, MY GET_TOI FUNCTION ACCOUNTS FOR THE FACT THAT THERE IS AN 'IMAGE' TOKEN
                    # IN THE PROMPT. DOES THIS PERSIST? OR IS THE TOKEN REMOVED? 
                    # IF THE TOKEN IS REMOVED, WE NEED TO ADD 1 TO EVERYTHING.
                    
                    # DOING AN ADDITIONAL +8 TO SKIP FORMATTING TOKENS: 
                    #  e.g. <s> USER: <image> \n
                    [(575 + tnum + 8, layername(model, layer, block_name, kind))], attn_head, knockout
                )
                row.append(scores)
            table.append(torch.stack(row))

    return torch.stack(table)


def calculate_hidden_flow(model, processor, constant_input, clean_input, corrupt_input, svo_tokens, mode, start, num_layers, block_name, kind=None, attn_head=None, knockout=None, correct_idx=None):

    low_score= trace_with_patch(model, processor, constant_input, clean_input, corrupt_input, svo_tokens=svo_tokens, mode=mode, states_to_patch=[], attn_head=attn_head, knockout=knockout)

    if block_name in ['text_encoder', 'text_decoder', 'vision_model', 'language_model']:
        scores= trace_important_states(
            model, processor, constant_input, clean_input, corrupt_input, svo_tokens, mode, start, num_layers, block_name, kind=kind, attn_head=attn_head, knockout=knockout, correct_idx=correct_idx
        )
    else:
        raise ValueError(f"Unknown block_name: {block_name}")

    if mode == 'image':
        text = constant_input
    else:
        text = clean_input

    return {
        "scores": scores,
        "low_score": low_score,
        "question": text,
        "correct_svo": processor.tokenizer.decode(svo_tokens[0]),
        "incorrect_svo": processor.tokenizer.decode(svo_tokens[1]),
        "block_name": block_name,
        "kind": kind,
        "attn_head": attn_head,
    }

def debug_hidden_flow(model, processor, df, task, block_name, kind, start, end, mode, attn_head=None, image_tensors=None):
    counter = 0
    temp_list = []
    outputs = []
    failed_rows = [] 
    
    for idx, row in df.iterrows():
        if mode == 'image':
            # This is for SVO probes where we store image tensors locally instead of file_paths
            if image_tensors:
                image_tensor1 = image_tensors.get(row["clean_image_path"])
                clean_input = to_pil_image(image_tensor1)
                image_tensor2 = image_tensors.get(row["corrupt_image_path"])
                corrupt_input = to_pil_image(image_tensor2)
                constant_input = row['clean_prompt']
            else:
                image_path1 = row["correct_image_path"]
                clean_input = Image.open(image_path1)
                image_path2 = row["incorrect_image_path"]
                corrupt_input = Image.open(image_path2)
                constant_input = row['clean_prompt']
    
        elif mode == 'text':
            if image_tensors:
                image_tensor = image_tensors.get(row["clean_image_path"])
                constant_input = to_pil_image(image_tensor)
            else:
                image_path = row["correct_image_path"]
                constant_input = Image.open(image_path)

            clean_input = row['clean_prompt'] 
            corrupt_input = row['corrupt_prompt']
        else:
            raise ValueError("Invalid mode specified.")

        # NOTE: WE DO NOT WANT TO USE THE CLEANED ANSWERS BC LLAVA IS CASE SENSITIVE.
        # Added the squeeze because tensors are returned as [1, num_tokens]
        correct_tokens = processor(text=row["generated_text"]).input_ids.squeeze()
        incorrect_tokens = processor(text=row["incorrect_answer"]).input_ids.squeeze()
        
        # GET THE TOKEN OF INTEREST
        matches, _ = generate_toi_matches(row, processor, task)  
        correct_idx = matches[row['correct_answer']]
    
        # CHANGE TO VAR NAME ANSWER TOKENS
        svo_tokens = [correct_tokens[-1], incorrect_tokens[-1]]        
        
        print(svo_tokens)

        try:
            temp = calculate_hidden_flow(model, processor, constant_input, 
                                         clean_input, corrupt_input, svo_tokens=svo_tokens, start=start, 
                                         num_layers=end, block_name=block_name, kind=kind, mode=mode, 
                                         attn_head=attn_head, correct_idx=correct_idx)
            temp_list.append(temp)
        except Exception as e:
            print(f"Error processing row {idx}: {str(e)}")
            counter += 1
            failed_rows.append(idx)  

    print("FINISHED PATCHING FOR KIND={}".format(str(kind)))
    print("FINISHED PATCHING FOR ATTN_HEAD={}".format(str(attn_head)))

    print("NUM EXCEPTIONS: ", counter)
    print("FAILED ROWS: ", failed_rows)  # Print the list of failed rows

    return temp_list
