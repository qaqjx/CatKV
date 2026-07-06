import hashlib
import json
import os

SPECIAL_TOKENS = "[##CATKV##]"

def serialize_and_hash(input_list):
    serialized_data = str(input_list.tolist()).encode('utf-8')
    hash_object = hashlib.md5(serialized_data)
    return hash_object.hexdigest()

def clean_json(data):
    def clean_string(s):
        return s.replace("#", "").strip() if isinstance(s, str) else s

    if isinstance(data, dict):  
        return {key: clean_json(value) for key, value in data.items()}
    elif isinstance(data, list):  
        return [clean_json(item) for item in data]
    else:  
        return clean_string(data)
    
def divide_prompt(prompt, special_tokens):
    if special_tokens in prompt:
        prefix_prompt = prompt.split(special_tokens)[0]
        question = prompt.split(special_tokens)[-1]
        doc_chunk_ids = prompt.split(special_tokens)[1:-1]

        doc_chunk_ids = [chunk for chunk in doc_chunk_ids if chunk != '']
        return prefix_prompt, doc_chunk_ids, question
    else:
        return "", [prompt]

def load_json_file(file_path, verbose=True):
    """
    Loads json of jsonline file
    """
    ext = os.path.splitext(file_path)[1].lower()
    if verbose:
        print("Loading file from: ", file_path)
    
    with open(file_path, 'r') as f:
        if ext == '.jsonl':
            return [json.loads(line) for line in f if line.strip()]
        elif ext == '.json':
            return json.load(f)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

def normalize_question(question):
    if not question.endswith("?"):
        question = question + "?"

    return question[0].lower() + question[1:]

def build_qa_prompt(example, query_prompt):
    q = normalize_question(example["question"])
    doc_prompts = [f"{ctx['title']}\n\n{ctx['text']}\n\n" for ctx in example["ctxs"]]
    #ex_prompt = f"{docs_text}\n\nBased on these texts, answer the question:\nQ: {q}\nA:"
    #q_prompt = f"\n\nAnswer the question based on the given passages. Answer the question within 5 words. Do NOT repeat the question or output any other words. Question: {q}\nAnswer:"
    q_prompt = f"{query_prompt}{q}\nAnswer:"
    return doc_prompts, q_prompt

def build_fewshot_prompt(example):
    q = "\n\n"+example["question"]
    doc_prompts = [f"{ctx['text']}" for ctx in example["ctxs"]]
    q_prompt = f"{q}"
    return doc_prompts, q_prompt

def normalize_context(s):
    s = SPECIAL_TOKENS + s
    return s

def combine_contexts(contexts):
    combined = ""
    for context in contexts:
        combined +=  SPECIAL_TOKENS + context + SPECIAL_TOKENS
    return combined

def truncate_context(context, max_model_len, tokenizer, manner="middle"):
    length = len(tokenizer.encode(context))
    if length > max_model_len:
        if manner == "middle":
            return context[:max_model_len // 2] + context[-max_model_len // 2:]
        else:
            raise ValueError(f"Invalid manner: {manner}")
    return context
    

def check_and_discard_contexts(contexts, max_model_len, tokenizer):
    truncated_ctxs = [truncate_context(context, max_model_len, tokenizer) for context in contexts]
    lengths = [len(tokenizer.encode(x)) for x in truncated_ctxs]

    while sum(lengths) > max_model_len:
        lengths.pop()

    return [truncated_ctxs[i] for i in range(len(lengths))]