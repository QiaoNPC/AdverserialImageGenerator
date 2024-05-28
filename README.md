# FGSM Brute Force Attack

This project implements a Fast Gradient Sign Method (FGSM) brute force attack on image classification models from Hugging Face. The script generates adversarial examples to fool the model into misclassifying the input images.


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

