# FGSM Brute Force Attack

This project demonstrates a **Fast Gradient Sign Method (FGSM)** brute force attack on image classification models hosted on Hugging Face 🤖. It generates **adversarial examples** by perturbing input images, attempting to force the model into misclassifying them.  

Supports targeted attacks by specifying a desired misclassification label, and allows testing across different epsilon values.

## Installation
```bash
pip install -r requirements.txt
```

## Usage
The script takes the following arguments:
```
  --model: The name of the Hugging Face model.
  --image: The path to the input image.
  --epsilon: The epsilon value for the attack (default: 0.05).
  --size: The size of the input image for the model.
  --target: The target class name.
  --d: The directory to save adversarial images.
```

