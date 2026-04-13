class Runtime:
    def __init__(self):
        self.params = None

    def get(self):
        if self.params is None:
            raise RuntimeError("Runtime no inicializado")
        return self.params


RUNTIME = Runtime()
