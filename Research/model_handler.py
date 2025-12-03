import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class ModelHandler:
    def __init__(self, model_name: str, quantize: bool = False):
        """
        Initialization of class arguments.

        1. model_name -> str -> Hugging face repo id.
        2. quantize -> bool -> Whether to quantize the model.
        """
        self.model_name = model_name
        self.quantize = quantize

    
    def load_model(self):
        """
        Loads the tokenizer and model according to the initialization parameters.

        Returns:
            model: The loaded AutoModelForCausalLM on the specified device.
            tokenizer: The corresponding AutoTokenizer.
        """
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=True)
            if self.quantize:
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    low_cpu_mem_usage=True,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    local_files_only=True
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        low_cpu_mem_usage=True,
                        device_map="auto",
                        local_files_only=True
                    )
        except OSError:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.quantize:
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    low_cpu_mem_usage=True,
                    torch_dtype=torch.float16,
                    device_map="auto",
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        low_cpu_mem_usage=True,
                        device_map="auto",
                    )
            
        return model, tokenizer
