models = {}

def register(name):
    def decorator(cls):
        models[name] = cls
        return cls
        
    return decorator


def make(model_spec, load_model=False):
    model = models[model_spec['name']](**model_spec['args'])
    
    if load_model:
        model.load_state_dict(model_spec['sd'])
    
    return model
