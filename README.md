# Foodie

Triplet net for food recognition

## Plan

1. Download Food-101 dataset
2. Create TripletImageFolder dataset (https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder, https://github.com/adambielski/siamese-triplet/blob/master/datasets.py#L79-L143)
3. Create the model
4. Create generic train() function and concrete main()


## Credits

Large portions of code taken from: https://github.com/adambielski/siamese-triplet/


## Tips

```
rsync -avzr -e ssh work/foodie user@ip:
```