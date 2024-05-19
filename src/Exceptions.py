class MissingFace(Exception):
    """Clase para una excepción personalizada."""
    def __init__(self, msg="Missing Face"):
        self.msg = msg
        super().__init__(self.msg)
