from .models import register, make, models
from . import transformer
from . import bottleneck
from . import loss
from . import larp_ar
from . import gptc
from . import larp_tokenizer


def get_model_cls(name):
    return models[name]