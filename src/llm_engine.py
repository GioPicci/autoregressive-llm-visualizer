import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.base_engine import BaseLLMEngine


class GPT2Engine(BaseLLMEngine):

    def __init__(self, model_name="gpt2"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            output_attentions=True
        ).to(self.device)

        self.model.eval()

    def tokenize(self, text):
        input_ids = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
        tokens = [self.tokenizer.decode([idx]) for idx in input_ids[0]]
        return input_ids, tokens

    def get_embeddings(self, input_ids):
        with torch.no_grad():
            embeddings = self.model.get_input_embeddings()(input_ids)

        return embeddings[0].cpu().numpy()

    def forward_pass(self, input_ids):

        with torch.no_grad():
            outputs = self.model(input_ids)

        logits = outputs.logits[0, -1, :]

        return logits, outputs.attentions

    def apply_sampling(self, logits, temperature=1.0, top_p=1.0):

        if temperature == 0.0:
            probs = torch.zeros_like(logits)
            probs[torch.argmax(logits)] = 1.0
            return probs.cpu().numpy()

        scaled_logits = logits / temperature
        probs = F.softmax(scaled_logits, dim=-1)

        if top_p < 1.0:

            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices[sorted_indices_to_remove]

            probs[indices_to_remove] = 0.0
            probs = probs / torch.sum(probs)

        return probs.cpu().numpy()

    def get_top_k_words(self, probs, k=50):

        top_k_indices = probs.argsort()[-k:][::-1]
        top_k_probs = probs[top_k_indices]

        top_k_words = [
            self.tokenizer.decode([idx]) for idx in top_k_indices
        ]

        return top_k_words, top_k_probs
