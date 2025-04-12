import os.path

from PIL.JpegImagePlugin import samplings

from Trainer import Trainer2D

DEFAULT_BATCH_SIZE = 256
BATCH_MESSAGE = "Insert batch size (press enter for 256):"
TRAINER_INITIALIZED = """Trainer initialized"""

class ConsoleInterface:


    def __init__(self, path = "../data/images/"):
        self.path = path
        self.epochs = 0

        self.running = True
        self.trainer = None
        self.is_trainer_initialized = False
        self.DEFAULT_LAYOUT = [(14, 8.0, 4), (16, 2.0, 8), (18, 0.5, 8), (18, 0.125, 4)]
        self.layout = self.DEFAULT_LAYOUT

    def run(self):
        while self.running:
            self.print_instructions()
            option = self.get_value("Select an option:","Invalid option",9)
            match option:
                case 1:
                    self.init_trainer()
                    self.epochs = 0
                case 2:
                    if self.is_trainer_initialized:
                        epochs = self.get_value("How many epochs (default 100)?",
                                                "Invalid value", 100)
                        self.epochs += epochs
                        self.trainer.train(epochs)
                    else:
                        self.new_line_string("Initialize trainer before running!")
                case 3:
                    if self.is_trainer_initialized:
                        self.trainer.reconstruct_image()
                case 4:
                    if self.is_trainer_initialized:
                        self.new_line_string("Insert filename (saved as jpeg:")
                        filename = input()
                        samples = self.get_value("Sampling factor (default is x4):",
                                                 "Invalid input", 4)
                        self.trainer.reconstruct_image(samples=samples, save_path=self.path+ filename +f"_{self.epochs}E_{samples}S{self.layout_to_string()}.jpeg")
                case 5:
                    self.trainer.train(1500)
                    self.trainer.reconstruct_image(samples=4,
                                   save_path=self.path + "colors" + f"_{3000}E_{4}S{self.layout_to_string()}.jpeg")

    def print_instructions(self):
        menu = f"""
                =========================================
                Instant-NGP Prototype 
                Trainer initialized - {self.is_trainer_initialized}
                =========================================
                1. Initialize trainer
                2. Train on image
                3. Reconstruct and display image
                4. Save image
                """
        print(menu)

    @staticmethod
    def new_line_string(message : str):
        print(message+"\n")



    def get_layout(self):
        valid = False
        while not valid:
            self.new_line_string(f"Insert layout, format is [(hash_size, cell_size, dimensions),..]"
                                 f" (default {self.DEFAULT_LAYOUT})")
            layout = input()
            if len(layout) == 0:
                return self.DEFAULT_LAYOUT
            else:
                try:
                    layout = eval(layout)
                except (SyntaxError,NameError):
                    pass
                if isinstance(layout, list):
                    for item in layout:
                        if isinstance(item, tuple) and len(item) == 3:
                            if all(isinstance(x, (int, float)) for x in item):
                                return layout
                self.new_line_string("Invalid layout.")
        return self.DEFAULT_LAYOUT


    def get_value(self,start_message : str, error_message : str, default : int):
        valid_option = False
        while not valid_option:
            self.new_line_string(start_message)
            value = input()
            if len(value) == 0:
                return default
            try:
                value = int(value)
                return value
            except ValueError:
                self.new_line_string(error_message)


    def check_file(self):
        valid = False
        while not valid:
            self.new_line_string("Input filename from " + self.path + " (with extension)")
            filename = input()
            full_path = self.path + filename
            if os.path.isfile(full_path):
                return full_path
            self.new_line_string("File does not exist in " + self.path)


    def layout_to_string(self) -> str:
        code = ""
        for h, c, d in self.layout:
            code += f"_{h}H-{c}C-{d}D"
        return code

    def init_trainer(self):
        path = self.check_file()

        batch_size = self.get_value(BATCH_MESSAGE, "Invalid batch size", 256)

        layout = self.get_layout()
        self.layout = layout


        self.assign_trainer(path, batch_size, layout)
        print(layout)

    def assign_trainer(self, path, batch_size, layout):
        self.trainer = Trainer2D(path, batch_size= batch_size, layout=layout)
        self.is_trainer_initialized = True
        self.new_line_string("Trainer initialized with given settings")