from src.communicator.communicator import Communicator
import serial

class CommunicatorUART(Communicator):
    """
        Class for communication via UART port. It needs to be opened on the device first.
    """
    def __init__(self, port, baudrate, parity, stopbits):
        """
            Initializes object and opens connection with it.

            Parameters:
            - port: str
                Address of the port in str type.
            - baudrate: int
                The measure of the number of changes to the signal (per second) that propagate through a transmission medium.
            - parity: str
                The parity bit.
            - stopbits: float
                Bits indicating the stop of the signal.

        """
        super().__init__()
        self.serial = serial.Serial(
            port=port,
            baudrate=baudrate,
            parity=parity,
            stopbits=stopbits,
            bytesize=serial.EIGHTBITS,
            xonxoff=False,
            timeout=1
        )


    def write(self, value):
        """
            Function writes value to the communication medium. 
        """
        for i in range(10):
            self.serial.write(value)


    def read(self):
        pass