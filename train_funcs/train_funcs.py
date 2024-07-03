funcs = {}

def register(name):
    def decorator(cls):
        funcs[name] = cls
        return cls
        
    return decorator


def make(config):
    func = funcs[config]

    return func