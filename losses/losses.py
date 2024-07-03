losses = {}

def register(name):
    def decorator(cls):
        losses[name] = cls
        return cls
        
    return decorator

def make(loss_spec, load_model=False):
    loss = losses[loss_spec['name']](**loss_spec['args'])
    
    return loss