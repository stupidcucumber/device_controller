class Controller():
    """
        This class is designed for connection with other device through UART.
    """
    def __init__(self,  communicator):
        """
            Initialize Controller with the object inherited from Communicator class.
        """
        self.communicator = communicator


    def perform_command(self, value):
        """
            Abstract method of performing command.
        """
        raise NotImplementedError("This is an abstract class! write method needs to be redefined.")
