import torch


def top_p_sampling(probabilities, top_p=0.9):
    """
    Top-p nucleus sampling.
    probabilities shape:
        (batch_size, vocab_size)
    return:
        next_token_idx shape: (batch_size, 1)
    """
    sorted_probabilities, sorted_indices = torch.sort(
        probabilities, dim=-1, descending=True
    )
    cumulative_probabilities = torch.cumsum(sorted_probabilities, dim=-1)

    mask = cumulative_probabilities > top_p
    mask[..., 1:] = mask[..., :-1].clone()
    mask[..., 0] = False

    sorted_probabilities = sorted_probabilities.masked_fill(mask, 0.0)

    denom = sorted_probabilities.sum(dim=-1, keepdim=True)
    denom = torch.clamp(denom, min=1e-12)
    sorted_probabilities = sorted_probabilities / denom

    sampled_idx_in_sorted = torch.multinomial(sorted_probabilities, num_samples=1)
    next_token_idx = torch.gather(sorted_indices, dim=-1, index=sampled_idx_in_sorted)

    return next_token_idx


def temperature_scaling(logits, temperature=1.0):
    """
    Temperature scaling on the last token logits.
    logits shape:
        (batch_size, seq_len, vocab_size)
    return:
        probabilities shape: (batch_size, vocab_size)
    """
    temperature = max(float(temperature), 1e-5)
    probabilities = torch.softmax(logits[:, -1, :] / temperature, dim=-1)
    return probabilities


def decode_token(
    input_tokens,
    model,
    max_tokens_to_generate,
    top_p=0.9,
    temperature=1.0,
    eos_token_id=None,
):
    """
    Autoregressive decoding.

    Args:
        input_tokens:
            shape can be:
            - (seq_len,)
            - (batch_size, seq_len)
        model:
            trained language model
        max_tokens_to_generate:
            number of new tokens to generate
        top_p:
            nucleus sampling threshold
        temperature:
            sampling temperature
        eos_token_id:
            optional end token id; if provided, stop when all batch items hit EOS

    Returns:
        generated tokens with shape:
            (batch_size, original_seq_len + generated_len)
    """
    model.eval()
    device = next(model.parameters()).device

    if not torch.is_tensor(input_tokens):
        input_tokens = torch.tensor(input_tokens, dtype=torch.long, device=device)
    else:
        input_tokens = input_tokens.to(device).long()

    if input_tokens.dim() == 1:
        input_tokens = input_tokens.unsqueeze(0)
    elif input_tokens.dim() != 2:
        raise ValueError(
            f"input_tokens should be 1D or 2D, but got shape {tuple(input_tokens.shape)}"
        )

    generated = input_tokens

    context_length = getattr(model, "context_length", None)

    with torch.no_grad():
        for _ in range(max_tokens_to_generate):
            if context_length is not None and generated.size(1) > context_length:
                model_input = generated[:, -context_length:]
            else:
                model_input = generated

            logits = model(model_input)
            probabilities = temperature_scaling(logits, temperature)
            next_token_idx = top_p_sampling(probabilities, top_p)

            generated = torch.cat([generated, next_token_idx], dim=-1)

            if eos_token_id is not None:
                if torch.all(next_token_idx.squeeze(-1) == eos_token_id):
                    break

    return generated
