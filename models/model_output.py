class ModelOutput:
    def __init__(self, irfs, fevd=None, forecast=None, meta=None):
        self.irfs = irfs
        self.fevd = fevd
        self.forecast = forecast
        self.meta = meta or {}