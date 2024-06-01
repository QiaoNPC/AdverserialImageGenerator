# Image Classification Misclassification Attack

## Goal

Can you trick the `Kaludi/food-category-classification-v2.0` model into misclassifying a *Seafood* as a *Fruit*?

## Usage

To achieve this result, you can use the following command:

```bash
python attack.py --model Kaludi/food-category-classification-v2.0 --image lab2/seafood.png --size 224 --target Fruit --d .
```