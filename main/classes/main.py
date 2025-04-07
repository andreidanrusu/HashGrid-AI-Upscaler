import Trainer

trainer = Trainer.Trainer2D("../data/images/lincoln.jpg")

trainer.initialize_trainer()

trainer.train(250)

trainer.reconstruct_image()