from .deepjit.warper import DeepJIT
from .tlel.warper import TLELModel
from .simcom.warper import SimCom

models = ["deepjit", "simcom", "lapredict", "tlel", "lr", "jitfine", "vccfinder"]

def init_model(model_name, language, device):   
    if  model_name == "deepjit":
        return DeepJIT(language=language, device=device)
    elif  model_name == "tlel":
        return TLELModel(language=language)
    elif model_name == "simcom":
        return SimCom(language=language, device=device)
    else:
        raise Exception("No such model")