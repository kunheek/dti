#!/bin/bash


# Copy datasets.
cp -r /workspace/kunkim/datasets/personalization/dreambooth data/dreambooth
cp -r /workspace/kunkim/datasets/personalization/dreambooth_mask data/dreambooth_mask
cp -r /workspace/kunkim/datasets/personalization/styledrop data/styledrop
cp -r /workspace/kunkim/datasets/personalization/ti data/ti

# Create output directory.
ln -sv /workspace/kunkim/dti_output outputs
