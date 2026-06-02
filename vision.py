"""Vision encoder integration for Qwen3-VL with deepstack support.

Runs HF vision encoder (eager), extracts main + deepstack features,
merges into text embeddings, and provides deepstack for decoder injection.
Supports both image and video inputs.
"""

import torch


IMAGE_TOKEN_ID = 151655
VIDEO_TOKEN_ID = 151656


def prepare_vision_embeds(hf_model, processor, inputs, device):
    """Run vision encoder and merge embeddings into text sequence.

    Handles image, video, or both. The same vision encoder processes
    both — video frames are treated as a batch of temporal patches.

    Returns:
        inputs_embeds: [1, seq_len, hidden_size] merged embeddings
        deepstack_embeds: list of [num_visual_tokens, hidden_size] tensors
        visual_mask: [seq_len] bool mask of all visual token positions
    """
    input_ids = inputs["input_ids"].to(device)
    pixel_values = inputs.get("pixel_values")
    image_grid_thw = inputs.get("image_grid_thw")
    pixel_values_videos = inputs.get("pixel_values_videos")
    video_grid_thw = inputs.get("video_grid_thw")

    visual = hf_model.model.visual if hasattr(hf_model.model, 'visual') else hf_model.visual
    if pixel_values is not None:
        pixel_values = pixel_values.to(dtype=visual.dtype, device=device)
    if image_grid_thw is not None:
        image_grid_thw = image_grid_thw.to(device)
    if pixel_values_videos is not None:
        pixel_values_videos = pixel_values_videos.to(dtype=visual.dtype, device=device)
    if video_grid_thw is not None:
        video_grid_thw = video_grid_thw.to(device)

    # Text embeddings
    inputs_embeds = hf_model.model.language_model.embed_tokens(input_ids)

    all_deepstack = []
    visual_mask = torch.zeros(input_ids.shape[-1], dtype=torch.bool, device=device)

    # Process images
    if pixel_values is not None and image_grid_thw is not None:
        with torch.no_grad():
            vision_output = visual(
                pixel_values, grid_thw=image_grid_thw, return_dict=True
            )

        image_features = vision_output.pooler_output
        deepstack_features = vision_output.deepstack_features

        image_mask = (input_ids[0] == IMAGE_TOKEN_ID)
        num_image_positions = image_mask.sum().item()

        if num_image_positions > 0 and image_features.shape[0] >= num_image_positions:
            inputs_embeds[0, image_mask] = image_features[:num_image_positions].to(inputs_embeds.dtype)
            visual_mask = visual_mask | image_mask

            if deepstack_features:
                all_deepstack = [
                    ds[:num_image_positions].to(inputs_embeds.dtype)
                    for ds in deepstack_features
                ]

    # Process video
    if pixel_values_videos is not None and video_grid_thw is not None:
        with torch.no_grad():
            video_output = visual(
                pixel_values_videos, grid_thw=video_grid_thw, return_dict=True
            )

        video_features = video_output.pooler_output
        video_deepstack = video_output.deepstack_features

        video_mask = (input_ids[0] == VIDEO_TOKEN_ID)
        num_video_positions = video_mask.sum().item()

        if num_video_positions > 0 and video_features.shape[0] >= num_video_positions:
            inputs_embeds[0, video_mask] = video_features[:num_video_positions].to(inputs_embeds.dtype)
            visual_mask = visual_mask | video_mask

            # Merge deepstack: if we already have image deepstack, combine
            if video_deepstack:
                if all_deepstack:
                    # Combine image + video deepstack at their respective positions
                    for idx in range(len(video_deepstack)):
                        combined = torch.zeros(
                            visual_mask.sum().item(), video_deepstack[idx].shape[-1],
                            dtype=inputs_embeds.dtype, device=device
                        )
                        # Map visual_mask positions back to image/video
                        vis_positions = visual_mask.nonzero(as_tuple=True)[0]
                        img_positions = (input_ids[0] == IMAGE_TOKEN_ID).nonzero(as_tuple=True)[0]
                        vid_positions = (input_ids[0] == VIDEO_TOKEN_ID).nonzero(as_tuple=True)[0]

                        # Scatter image deepstack
                        for j, pos in enumerate(img_positions):
                            idx_in_vis = (vis_positions == pos).nonzero(as_tuple=True)[0]
                            if len(idx_in_vis) > 0 and j < all_deepstack[idx].shape[0]:
                                combined[idx_in_vis[0]] = all_deepstack[idx][j]
                        # Scatter video deepstack
                        for j, pos in enumerate(vid_positions):
                            idx_in_vis = (vis_positions == pos).nonzero(as_tuple=True)[0]
                            if len(idx_in_vis) > 0 and j < video_deepstack[idx].shape[0]:
                                combined[idx_in_vis[0]] = video_deepstack[idx][j]

                        if idx < len(all_deepstack):
                            all_deepstack[idx] = combined
                        else:
                            all_deepstack.append(combined)
                else:
                    all_deepstack = [
                        ds[:num_video_positions].to(inputs_embeds.dtype)
                        for ds in video_deepstack
                    ]

    deepstack_embeds = all_deepstack if all_deepstack else None
    visual_mask_out = visual_mask if visual_mask.any() else None

    return inputs_embeds, deepstack_embeds, visual_mask_out
