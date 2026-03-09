class BaseLLMEngine:
    def tokenize(self,text):
        raise NotImplementedError

    def get_embeddings(self,input_ids):
        raise NotImplementedError

    def forward_pass(self,input_ids):
        raise NotImplementedError

    def apply_sampling(self,logits,temperature=1.0,top_p=1.0):
        raise NotImplementedError

    def get_top_k_words(self,probs,k=50):
        raise NotImplementedError