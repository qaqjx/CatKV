import torch
import time

# Only support the greedy search for now
class Request:
    def __init__(self, model, tokenizer, device):
        model.eval()
        self.model = model
        self.tokenizer = tokenizer
        self.past_kv = None 
        self.device = device

    def clear(self):
        self.model.is_blend = 0
        self.past_kv = None
        torch.cuda.empty_cache()

    def _process_texts(self, input_text):
        model_inputs = {}
        input_ids = self.tokenizer.encode(input_text)

        model_inputs["input_ids"] = input_ids
        model_inputs["attention_mask"] = [1] * len(model_inputs["input_ids"])

        for key in model_inputs:
            model_inputs[key] = torch.tensor(model_inputs[key]).int().unsqueeze(0).to(self.device)

        return model_inputs

    def generate(self, text=None, input_ids=None, **kwargs):
        if input_ids is None:
            model_inputs = self._process_texts(text)
            input_ids = model_inputs['input_ids']

        with torch.no_grad():
            result = self.inference(input_ids, **kwargs)
        return result

    def inference(self, input_ids, max_new_length: int = 100, device="cuda"):
        start = time.time()

        if input_ids.dim() == 1:
            input_ids = input_ids[None, :]
        input_ids = input_ids.to(device)
        attention_mask = torch.ones_like(input_ids)
        assert input_ids.size(0) == 1 # batch size is 1
        
        length = input_ids.size(1)
        end_token_ids = [self.tokenizer.eos_token_id]
        logits = None
        past_key_values = self.past_kv

        self.model.model.blend_meta = self.model.blend_meta
        ttft = 0
        # prefill phase
        for i in range(max_new_length + 1):
            if i == 0:
                # prefill phase   
                out = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=True,
                    return_dict=True,
                    past_key_values=past_key_values
                )
                
                logits, past_key_values = out.logits, out.past_key_values
                self.model.model.blend_meta["phase"] = "decode"
            else:
                if i == 1:
                    end = time.time()
                    ttft = end - start                
                # decode phase
                out = self.model(
                    input_ids=input_ids[:, -1:],
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True
                )
                logits, past_key_values = out.logits, out.past_key_values

            if i == 0 and self.model.blend_meta["state"] == "store":
                past_key_values.save_all_kv_tensors(self.model.blend_meta["hash_text"] ,self.model.blend_meta["indices"])

            # TODO : handle logits for different sample types               
            logits = logits[:, -1, :]
            word = logits.argmax(dim=-1)    

            if word.item() in end_token_ids or i == max_new_length:
                break

            input_ids = torch.cat((input_ids, word.view(1, 1)), dim=-1)
            attention_mask = torch.cat(
                (attention_mask, torch.ones((attention_mask.size(0), 1), dtype=torch.int, device=attention_mask.device)),
                dim=-1
            )

        self.past_kv = past_key_values

        return [self.tokenizer.decode(input_ids.squeeze(0)[length:])], ttft