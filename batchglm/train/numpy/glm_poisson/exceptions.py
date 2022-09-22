class NoScaleError(Exception):
    """
    Exception raised for attempting to access the scale parameter (or one of its derived methods) of a poisson model.
    """

    def __init__(self, method: str):
        self.message = f"Attempted to access {method}. No scale parameter is fit for poisson - please use location."
        super().__init__(self.message)
