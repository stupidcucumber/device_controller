class Communicator():
    """
        An interface for implementing Communication  between devices.
    """
    def __init__(self):
        pass


    def write(self, value):
        raise NotImplementedError("write method needs to be redefined in the inherited class.")
    
    
    def read(self):
        raise NotImplementedError("write method needs to be redefined in the inherited class.")
