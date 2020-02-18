import numpy as np

def remove_misclassified(model, images, labels):
    predictions = model.predict(images)
    predicted_labels = np.argmax(predictions, axis=-1)
    
    correct_label = np.equal(predicted_labels, labels)

    return images[correct_label], labels[correct_label]

# ART come considera i failed?
def remove_failed(images, labels, adversarials):
    successful = np.array([x is not None for x in adversarials])
    successful_adversarials = np.array([x for x in adversarials if x is not None])
    return images[successful], labels[successful], successful_adversarials