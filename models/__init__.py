from .wav2vec import Wav2Vec2Model

model_list = {
    'facebook/wav2vec2-base-960h': Wav2Vec2Model
}

def load_model(model_name):
    return model_list[model_name]
