from PIL import Image
import torch
from transformers import AutoModelForImageClassification, AutoImageProcessor
from torch import nn
from torchvision import transforms
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def fgsm_brute_force(hugging_face_model, image_path, epsilon, model_image_size, target_class_name, directory): 
    predicted_classes = set()

    model = AutoModelForImageClassification.from_pretrained(hugging_face_model)
    processor = AutoImageProcessor.from_pretrained(hugging_face_model)
    model.eval()

    continue_brute_force = True
    i = 0

    while continue_brute_force:
        target_class = torch.tensor([i]) 

        original_image, initial_prediction, adversarial_image, final_prediction = create_adversarial_example(image_path, target_class, epsilon, model_image_size, model, directory)

        to_pil_image = transforms.ToPILImage()

        to_pil_image(adversarial_image.cpu().detach().squeeze(0)).save(f"{directory}/adversarial_image{i}.png")

        image = Image.open(f"{directory}/adversarial_image{i}.png")
        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
        print(f"{predicted_class_idx}: {model.config.id2label[predicted_class_idx]}")

        predicted_classes.add(model.config.id2label[predicted_class_idx])
        
        if target_class_name.lower() not in model.config.id2label[predicted_class_idx].lower():
            os.remove(f"{directory}/adversarial_image{i}.png")
        else:
            continue_brute_force = False
            break

        i += 1

    print(predicted_classes)

def fast_gradient_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)

    return perturbed_image

def create_adversarial_example(image_path, target_class, epsilon, model_image_size, model, directory):
    image = Image.open(f"{directory}/{image_path}")
    if image.mode == 'RGBA':
        image = image.convert('RGB')

    preprocess = transforms.Compose([
        transforms.Resize(model_image_size),
        transforms.ToTensor(),
    ])
    image = preprocess(image).unsqueeze(0) 
    image = image.to(device) 

    image.requires_grad = True

    output = model(image)
    initial_prediction = torch.argmax(output.logits, dim=1).item()
    loss = nn.CrossEntropyLoss()(output.logits, target_class)
    model.zero_grad()
    loss.backward()
    data_grad = image.grad.data
    perturbed_image = fast_gradient_attack(image, epsilon, data_grad)
    output = model(perturbed_image)
    predicted_class_idx = output.logits.argmax(-1).item()
    final_prediction = predicted_class_idx

    return image.squeeze(0), initial_prediction, perturbed_image.squeeze(0), final_prediction