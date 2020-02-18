import torchvision
import numpy as np
import foolbox

batch_size = 1

torch_model = torchvision.models.resnet50(pretrained=True)
torch_model.eval()

# Prepare the image and the label
image, label = foolbox.utils.imagenet_example()
image = np.moveaxis(image, 2, 0)
image = image / 255
label = np.array(label)

# Create a fake batch
images = np.repeat(image[np.newaxis], batch_size, axis=0)
labels = np.repeat(label[np.newaxis], batch_size, axis=0)

# Create a PyTorchModel
mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
stdevs = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
model = foolbox.models.PyTorchModel(torch_model, bounds=(0, 1), num_classes=1000, preprocessing=(mean, stdevs))

# Compute the gradients
grads = model.gradient(images, labels)

print(grads[0].mean())