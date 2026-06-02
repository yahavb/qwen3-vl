"""Vision encoder integration for Qwen3-VL.

Uses the HF model's own vision pipeline to produce merged embeddings,
then hands off to our custom decoder for generation.
"""

import torch


IMAGE_TOKEN_ID = 151655
VIDEO_TOKEN_ID = 151656


def prepare_vision_embeds(hf_model, processor, inputs, device):
    """Use HF model internals to produce merged text+vision embeddings.

    This calls the HF model's embedding + vision merge logic (which handles
    the vision encoder, patch merging, and token scattering), then returns
    the merged inputs_embeds ready for our custom decoder.

    Args:
        hf_model: Full HF Qwen3VLForConditionalGeneration model (on device)
        processor: HF processor (for reference)
        inputs: Dict from processor() with input_ids, pixel_values, image_grid_thw etc.
        device: Neuron device

    Returns:
        inputs_embeds: [1, seq_len, hidden_size] merged embeddings
    """
    input_ids = inputs["input_ids"].to(device)
    pixel_values = inputs.get("pixel_values")
    image_grid_thw = inputs.get("image_grid_thw")

    if pixel_values is not None:
        pixel_values = pixel_values.to(device)
    if image_grid_thw is not None:
        image_grid_thw = image_grid_thw.to(device)

    with torch.no_grad():
        # Get text embeddings
        inputs_embeds = hf_model.model.language_model.embed_tokens(input_ids)

        # Run vision encoder and merge if we have image inputs
        if pixel_values is not None and image_grid_thw is not None:
            # Call the model's vision forward to get image features
            image_outputs = hf_model.model.visual(
                pixel_values,
                grid_thw=image_grid_thw,
            )
            # The visual model returns the merged/projected features
            if hasattr(image_outputs, 'last_hidden_state'):
                image_features = image_outputs.last_hidden_state
            else:
                image_features = image_outputs

            # Scatter image features into embed positions
            image_mask = (input_ids[0] == IMAGE_TOKEN_ID)
            num_image_positions = image_mask.sum().item()
            if num_image_positions > 0:
                inputs_embeds[0, image_mask] = image_features[:num_image_positions].to(inputs_embeds.dtype)

    return inputs_embeds
