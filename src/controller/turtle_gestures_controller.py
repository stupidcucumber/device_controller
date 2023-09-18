from src.controller import controller
from bitarray import bitarray

class TurtleGesturesController(controller.Controller):
    """
        Class for operating turtle.
        We have the following command_id's:
            -1, means that no command is passed. Turtle needs to stand still.
             2, means move forward
             1, means stop,
             4, means back,
             6, means turn left
             7, means turn right
    """
    def __init__(self, communicator, speed=100):
        """
            Initializes the controller of the Turtle. Accpets object of type Communicator.
            - communicator: Communicator:
                Communicator which will be communicate with the turtle.
            - speed: int
                The speed of the movement of the turtle.   
        """
        super().__init__(communicator)

        self.speed = speed
    

    def perform_command(self, command):
        """
            - param command: 'forward','back','left','right','stop'
            - return: bytes
        """
        move_array = [2, 4, 6, 7]
        stop_message = b'\xff\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'

        speed_on_x = 0 if (command == 2 or command == 4) else self.speed
        speed_on_y = self.speed
        
        dir_y = 1 if command == 2 else 0

        if command == 6:
            dir_x = 1
        elif command == 7:
            dir_x = 0

        speed_direction = int.from_bytes(bitarray([0, 0, 0, 0, 0, 0, dir_y, dir_x]), "little")

        message = [255, 0, speed_direction, speed_on_x, speed_on_y, 0, 0, 0, 0, 0, 0]
        src = sum(message[1:]) % 256
        message.append(src)
        bts = stop_message if (command not in move_array) else bytes(message)

        self.communicator.write(bts)