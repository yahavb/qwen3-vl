"""Vision encoder integration for Qwen3-VL with deepstack support.

Runs HF vision encoder (eager), extracts main + deepstack features,
merges into text embeddings, and provides deepstack for decoder injection.
"""

import torch


IMAGE_TOKEN_ID = 151655
VIDEO_TOKEN_ID = 151656


def prepare_vision_embeds(hf_model, processor, inputs, device):
    """Run vision encoder and merge embeddings into text sequence.

    Returns:
        inputs_embeds: [1, seq_len, hidden_size] merged embeddings
        deepstack_embeds: list of [num_visual_tokens, hidden_size] tensors
            to inject after decoder layers 0, 1, 2 (corresponding to vision
            encoder layers 8, 16, 24)
        visual_mask: [seq_len] bool mask of visual token positions
    """
    input_ids = inputs["input_ids"].to(device)
    pixel_values = inputs.get("pixel_values")
    image_grid_thw = inputs.get("image_grid_thw")

    visual = hf_model.model.visual if hasattr(hf_model.model, 'visual') else hf_model.visual
    if pixel_values is not None:
        pixel_values = pixel_values.to(dtype=visual.dtype, device=device)
    if image_grid_thw is not None:
        image_grid_thw = image_grid_thw.to(device)

    # Text embeddings
    inputs_embeds = hf_model.model.language_model.embed_tokens(input_ids)

    deepstack_embeds = None
    visual_mask = None

    if pixel_values is not None and image_grid_thw is not None:
        with torch.no_grad():
            # Run vision encoder — returns BaseModelOutputWithDeepstackFeatures
            vision_output = visual(
                pixel_values, grid_thw=image_grid_thw, return_dict=True
            )

        # Main image embeddings (after final merger + projection)
        image_features = vision_output.pooler_output  # [num_visual_tokens, out_hidden_size]

        # Deepstack features from intermediate layers
        deepstack_features = vision_output.deepstack_features  # list of [num_visual_tokens, out_hidden_size]

        # Build visual position mask
        image_mask = (input_ids[0] == IMAGE_TOKEN_ID)
        num_image_positions = image_mask.sum().item()

        if num_image_positions > 0 and image_features.shape[0] >= num_image_positions:
            # Scatter main embeddings into text at image token positions
            inputs_embeds[0, image_mask] = image_features[:num_image_positions].to(inputs_embeds.dtype)

            # Prepare deepstack for decoder injection
            if deepstack_features:
                deepstack_embeds = [
                    ds[:num_image_positions].to(inputs_embeds.dtype)
                    for ds in deepstack_features
                ]

            visual_mask = image_mask

    return inputs_embeds, deepstack_embeds, visual_mask
