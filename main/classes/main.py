import Trainer
data_folder = "../data/images/"
file_name = "empire_state_128"
trainer = Trainer.Trainer2D(data_folder + file_name + ".png", 256)
epochs = 500
trainer.train(epochs)

trainer.reconstruct_image(data_folder+file_name+"_reconstructed_"+str(epochs)+".png")