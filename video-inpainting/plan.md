Modifications
1. Data loading: image, caption, (mask, prev_image, prev_mask)
2. Model: LP[ENC(prev_image * prev_mask), ENC(image * -mask)] + input_feature
3. Optimizer: Lora + LP
4. Sampling: add inputs