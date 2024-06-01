# Image Classification Misclassification Attack

## Goal

Can you trick the `google/mobilenet_v2_1.0_224` model into misclassifying a *German Shepherd* as a *Malinois*?

## Usage

To achieve this result, you can use the following command:

```bash
python attack.py --model google/mobilenet_v2_1.0_224 --image lab1/GermanShepherd.png --size 224 --target malinois --d .
```