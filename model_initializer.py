# model_initializer.py
from transformers import SeamlessM4Tv2Model, AutoProcessor
import torch

def initialize_model():
    processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
    model = SeamlessM4Tv2Model.from_pretrained("facebook/seamless-m4t-v2-large")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    return model, processor, device
