"""Vision encoder integration for Qwen3-VL.

Runs HF vision encoder (eager, on Neuron) and merges visual embeddings
into the text token sequence at image placeholder positions.
"""

import torch


IMAGE_TOKEN_ID = 151655
VIDEO_TOKEN_ID = 151656


def get_vision_embeddings(hf_model, pixel_values, image_grid_thw, video_grid_thw=None, pixel_values_videos=None):
    """Run the HF vision encoder to get visual embeddings.

    Args:
        hf_model: Full HF Qwen3VLForConditionalGeneration model (on device)
        pixel_values: Image pixel values from processor
        image_grid_thw: Image grid dimensions from processor
        video_grid_thw: Video grid dimensions (optional)
        pixel_values_videos: Video pixel values (optional)

    Returns:
        image_embeds: [num_image_tokens, hidden_size] or None
        video_embeds: [num_video_tokens, hidden_size] or None
    """
    image_embeds = None
    video_embeds = None

    if pixel_values is not None and image_grid_thw is not None:
        with torch.no_grad():
            image_embeds = hf_model.model.get_image_features(
                pixel_values=pixel_values,
                grid_thw=image_grid_thw,
            )
            if hasattr(image_embeds, 'last_hidden_state'):
                image_embeds = image_embeds.last_hidden_state

    if pixel_values_videos is not None and video_grid_thw is not None:
        with torch.no_grad():
            video_embeds = hf_model.model.get_video_features(
                pixel_values_videos=pixel_values_videos,
                grid_thw=video_grid_thw,
            )
            if hasattr(video_embeds, 'last_hidden_state'):
                video_embeds = video_embeds.last_hidden_state

    return image_embeds, video_embeds


def merge_vision_embeddings(embed_tokens, input_ids, image_embeds=None, video_embeds=None, device=None):
    """Merge vision embeddings into text token embeddings.

    Replaces IMAGE_TOKEN_ID positions with image_embeds and
    VIDEO_TOKEN_ID positions with video_embeds.

    Args:
        embed_tokens: nn.Embedding layer (on device)
        input_ids: [1, seq_len] token IDs
        image_embeds: [num_image_tokens, hidden_size] or None
        video_embeds: [num_video_tokens, hidden_size] or None
        device: target device

    Returns:
        inputs_embeds: [1, seq_len, hidden_size] merged embeddings
    """
    inputs_embeds = embed_tokens(input_ids)  # [1, seq_len, hidden]

    if image_embeds is not None:
        image_mask = (input_ids[0] == IMAGE_TOKEN_ID)
        num_image_positions = image_mask.sum().item()
        if num_image_positions > 0 and image_embeds.shape[0] >= num_image_positions:
            inputs_embeds[0, image_mask] = image_embeds[:num_image_positions].to(inputs_embeds.dtype)

    if video_embeds is not None:
        video_mask = (input_ids[0] == VIDEO_TOKEN_ID)
        num_video_positions = video_mask.sum().item()
        if num_video_positions > 0 and video_embeds.shape[0] >= num_video_positions:
            inputs_embeds[0, video_mask] = video_embeds[:num_video_positions].to(inputs_embeds.dtype)

    return inputs_embeds
