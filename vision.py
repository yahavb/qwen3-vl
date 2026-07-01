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

        # On Neuron the vision path is a static-shape compiled graph, so
        # video_features.shape[0] is fixed (e.g. 1152) regardless of the clip's
        # real grid, while num_video_positions tracks the processor's per-clip
        # <video> placeholder count (e.g. 683/692). When they differ, decode
        # crashes: model_fused adds deepstack rows onto hidden[0, visual_mask],
        # so deepstack rows MUST equal visual_mask positions. We therefore fill
        # exactly n = min(features, placeholders) positions and truncate the
        # main features AND the deepstack to that same n, keeping every count
        # consistent. If features < placeholders, the surplus placeholders stay
        # as text embeds (not marked visual); if features > placeholders, the
        # extra features are dropped.
        n_fill = min(int(video_features.shape[0]), int(num_video_positions))
        if n_fill > 0:
            vid_pos = video_mask.nonzero(as_tuple=True)[0][:n_fill]
            fill_mask = torch.zeros_like(video_mask)
            fill_mask[vid_pos] = True
            inputs_embeds[0, fill_mask] = video_features[:n_fill].to(inputs_embeds.dtype)
            visual_mask = visual_mask | fill_mask

            # Deepstack for the video positions, truncated to the SAME n_fill so
            # deepstack row count == visual_mask positions at decode time.
            if video_deepstack:
                if all_deepstack:
                    # Image + video in one request: rebuild each deepstack level
                    # sized to the full visual_mask, scattering image rows into
                    # image positions and video rows into the filled video ones.
                    vis_positions = visual_mask.nonzero(as_tuple=True)[0]
                    img_positions = (input_ids[0] == IMAGE_TOKEN_ID).nonzero(as_tuple=True)[0]
                    for idx in range(len(video_deepstack)):
                        combined = torch.zeros(
                            vis_positions.shape[0], video_deepstack[idx].shape[-1],
                            dtype=inputs_embeds.dtype, device=device
                        )
                        pos_to_row = {int(p): r for r, p in enumerate(vis_positions.tolist())}
                        for j, pos in enumerate(img_positions.tolist()):
                            r = pos_to_row.get(pos)
                            if r is not None and idx < len(all_deepstack) and j < all_deepstack[idx].shape[0]:
                                combined[r] = all_deepstack[idx][j]
                        for j, pos in enumerate(vid_pos.tolist()):
                            r = pos_to_row.get(pos)
                            if r is not None and j < video_deepstack[idx].shape[0]:
                                combined[r] = video_deepstack[idx][j].to(inputs_embeds.dtype)
                        if idx < len(all_deepstack):
                            all_deepstack[idx] = combined
                        else:
                            all_deepstack.append(combined)
                else:
                    # Video only (the captioning case): truncate to n_fill.
                    all_deepstack = [
                        ds[:n_fill].to(inputs_embeds.dtype)
                        for ds in video_deepstack
                    ]

    deepstack_embeds = all_deepstack if all_deepstack else None
    visual_mask_out = visual_mask if visual_mask.any() else None

    return inputs_embeds, deepstack_embeds, visual_mask_out
