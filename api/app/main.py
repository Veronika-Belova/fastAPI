import PIL
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from utils.model_func import class_id_to_label, load_model, transform_image
from utils.model_bert_func import load_bert_model, predict_sentiment

model = None
bert_model = None
tokenizer = None
device = 'cpu'
app = FastAPI()

class ImageClass(BaseModel):
    prediction: str

class TextClass(BaseModel):
    text: str

@app.on_event("startup")
def startup_event():
    global model
    model = load_model()

@app.get('/')
def return_info():
    return 'Hello FastAPI'

@app.post('/classify')
def classify(file: UploadFile = File(...)):
    image = PIL.Image.open(file.file)
    adapted_image = transform_image(image)
    pred_index = model(adapted_image.unsqueeze(0)).detach().cpu().numpy().argmax()
    imagenet_class = class_id_to_label(pred_index)
    response = ImageClass(
        prediction=imagenet_class
    )
    return response

@app.on_event("startup")
def startup_event():
    global bert_model, tokenizer, device
    bert_model, tokenizer, device = load_bert_model()


@app.post('/clf_text')
def clf_text(data: TextClass):
    sentiment = predict_sentiment(data.text, tokenizer, device, bert_model)
    return {"sentiment": sentiment}
