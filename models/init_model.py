models = ["deepjit", "simcom", "lapredict", "tlel", "lr", "jitfine"]

def init_model(model_name, language, device):   
    if model_name == "deepjit":
        from .deepjit.warper import DeepJIT
        return DeepJIT(language=language, device=device)
    elif model_name == "tlel":
        from .tlel.warper import TLELModel
        return TLELModel(language=language)
    elif model_name == "simcom":
        from .simcom.warper import SimCom
        return SimCom(language=language, device=device)
    elif model_name == "lapredict":
        from .lapredict.warper import LAPredict
        return LAPredict(language=language)
    elif model_name == "lr":
        from .lr.warper import LogisticRegression
        return LogisticRegression(language=language)
    elif model_name == "jitfine":
        from .jitfine.warper import JITFine
        return JITFine(language=language, device=device)
    else:
        raise Exception("No such model")