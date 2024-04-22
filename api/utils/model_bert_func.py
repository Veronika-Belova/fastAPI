import torch
import torchvision.transforms as T
from transformers import AutoModel, AutoTokenizer
from torch import nn

def load_bert_model():
    class BERTClassifier(nn.Module):
        def __init__(self, bert_path="cointegrated/rubert-tiny2"):
            super().__init__()
            self.bert = AutoModel.from_pretrained(bert_path)
            for param in self.bert.parameters():
                param.requires_grad = False
            self.linear = nn.Sequential(
                nn.Linear(312, 150),
                nn.Dropout(0.1),
                nn.ReLU(),
                nn.Linear(150, 1),
                nn.Sigmoid()
            )
        def forward(self, x, masks):
            bert_out = self.bert(x, attention_mask=masks)[0][:, 0, :]
            out = self.linear(bert_out)
            return out
    
    tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2")
    model = BERTClassifier()
    device = 'cpu'
    model.to(device)
    model.load_state_dict(torch.load('utils/BERTmodel_weights2.pth', map_location='cpu'))
    model.eval()
    return model, tokenizer, device

def predict_sentiment(text, tokenizer, device, model):
    MAX_LEN = 100
    encoded_review = tokenizer.encode_plus(
        text,
        max_length=MAX_LEN,
        add_special_tokens=True,
        return_token_type_ids=False,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    input_ids = encoded_review['input_ids'].to(device)
    attention_mask = encoded_review['attention_mask'].to(device)

    with torch.no_grad():
        output = model(input_ids, attention_mask)
        prediction = torch.round(output).cpu().numpy()[0][0]
    if prediction == 1:
        return "Позитивный отзыв 😀"
    else:
        return "Негативный отзыв 😟"
