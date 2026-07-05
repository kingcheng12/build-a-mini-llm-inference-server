"""
Build a Mini LLM Inference Server

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - stable_softmax
import numpy as np
def stable_softmax(logits):
    # TODO: compute a numerically stable softmax over the last axis of logits.
    logits_shifted = logits - np.max(logits, axis = -1, keepdims=True)
    exp_logits = np.exp(logits_shifted)

    return exp_logits / np.sum(exp_logits, axis = -1, keepdims=True)

# Step 2 - apply_temperature
def apply_temperature(logits, temperature):
    # TODO: scale logits by 1 / temperature; if temperature <= 0, return logits unchanged (greedy).
    
    if temperature > 0:
        return logits / temperature
    else:
        return logits

# Step 3 - top_k_filter
import numpy as np

def top_k_filter(logits, k):
    """Mask logits outside the top-k per row to -inf."""
    # TODO: keep only the k largest logits along the last axis, set the rest to -inf
    logits = np.asarray(logits)
    vocab_size = logits.shape[-1]

    if k <= 0:
        return np.full(logits.shape, -np.inf, dtype=float)

    if k >= vocab_size:
        return logits.astype(float)

    threshold = np.partition(logits, -k, axis=-1)[..., -k:][..., 0:1]
    mask = logits >= threshold

    return np.where(mask, logits, -np.inf)

# Step 4 - top_p_filter
def top_p_filter(logits, p):
    # TODO: keep smallest set of tokens whose cumulative prob >= p, mask the rest to -inf.

    if p >= 1:
        return logits
    
    probs = stable_softmax(logits)
    desc_idx = np.argsort(-probs, axis=-1)
    sorted_probs = np.take_along_axis(probs, desc_idx, axis = -1)
    cumsum_probs = np.cumsum(sorted_probs, axis = -1)
    cutoff_idx = np.argmax(cumsum_probs >= p, axis = -1)

    # Mask in sorted order
    positions = np.arange(logits.shape[-1])
    keep_sorted = positions <= cutoff_idx[..., None]

    # Scatter mask back to original order
    keep_original = np.zeros_like(keep_sorted, dtype=bool)
    np.put_along_axis(keep_original, desc_idx, keep_sorted, axis=-1)

    return np.where(keep_original, logits, -np.inf)

# Step 5 - sample_from_probs
def sample_from_probs(probs, rng):
    # TODO: draw a single token id from the categorical distribution probs using rng
    
    return rng.choice(len(probs), p=probs)

# Step 6 - greedy_select
def greedy_select(logits):
    # TODO: return the index of the maximum logit (ties -> lowest index).
    
    return np.argmax(logits, axis=-1)

# Step 7 - build_vocab
def build_vocab(corpus, special_tokens):
    # TODO: build a character-level vocab; specials get the lowest ids, then sorted unique chars.
    id_to_token = special_tokens
    all_chars = set()
    for text in corpus:
        for c in text:
            all_chars.add(c)

    all_chars = sorted(list(all_chars))
    id_to_token.extend(all_chars)

    token_to_id = {token:i for i, token in enumerate(id_to_token)}

    return {'token_to_id':token_to_id, 'id_to_token':id_to_token}

# Step 8 - encode_prompt
def encode_prompt(text, vocab, add_bos=True):
    # TODO: encode text into token ids using vocab, optionally prepending <bos>.
    
    ids = []
    if add_bos:
        ids.append(vocab['token_to_id']['<bos>'])
    
    for token in text:
        if token in vocab['token_to_id']:
            ids.append(vocab['token_to_id'][token])
        else:
            ids.append(vocab['token_to_id']['<unk>'])

    return ids

# Step 9 - decode_tokens
def decode_tokens(token_ids, vocab, skip_special=True):
    # TODO: convert token ids back into a string using vocab['id_to_token'], optionally skipping specials.
    
    out = []
    max_special_idx = 0
    for token in vocab['token_to_id']:
        if token[0] == '<' and token[-1] == '>':
            max_special_idx += 1
        else:
            break

    for token_id in token_ids:
        if token_id < max_special_idx and skip_special: # is special
            continue
        else:
            out.append(vocab['id_to_token'][token_id])
    
    return ''.join(out)

# Step 10 - embed_tokens
import numpy as np

def embed_tokens(token_ids, embedding_matrix):
    # TODO: return the (T, D) embedding rows for each token id in token_ids
    return np.array([embedding_matrix[token_id] for token_id in token_ids])

# Step 11 - linear_projection
import numpy as np

def linear_projection(x, weight, bias=None):
    # TODO: Apply y = x @ weight + bias, with bias optional and broadcasting over leading axes.
    x = np.array(x)
    weight = np.array(weight)
    outdim = weight.shape[-1]

    y = x @ weight
    if bias is not None:
        bias = np.array(bias)
        # change shape of bias to enable broadcast
        bias = bias.reshape(outdim,)
        y += bias

    return y

# Step 12 - init_kv_cache
import numpy as np

def init_kv_cache(max_seq_len, d_model):
    # TODO: allocate empty K and V buffers and a length counter for a single sequence
    
    cache = {}
    cache['K'] = np.zeros((max_seq_len, d_model), dtype = np.float32)
    cache['V'] = np.zeros((max_seq_len, d_model), dtype = np.float32)
    cache['length'] = 0

    return cache

# Step 13 - append_kv
import numpy as np

def append_kv(cache, k_new, v_new):
    # TODO: write k_new and v_new into the cache starting at cache['length'] and bump length.
    
    current_length = cache['length']
    len_new = k_new.shape[0]
    cache['K'][current_length:current_length+len_new] = k_new
    cache['V'][current_length:current_length+len_new] = v_new
    cache['length'] += len_new

    return cache

# Step 14 - causal_attention
import numpy as np

def causal_attention(q, k, v, is_causal=True):
    # TODO: scaled dot-product attention with optional causal mask, returns (Tq, D)

    d_model = q.shape[-1]
    scores = q @ k.T / np.sqrt(d_model)

    q_len = scores.shape[-2]
    k_len = scores.shape[-1]

    if is_causal:
        mask = np.tril(np.ones((q_len, k_len), dtype=bool), k=k_len - q_len)
        scores = np.where(mask, scores, -np.inf)
    
    attn_probs = stable_softmax(scores)

    out = attn_probs @ v
    return out

# Step 15 - model_prefill
def model_prefill(token_ids, params):
    # TODO: embed tokens, project Q/K/V, fill the KV cache, run causal attention, return last-position logits.
    token_ids = np.array(token_ids)
    emb = embed_tokens(token_ids, params["embedding"])

    Q = linear_projection(emb, params['Wq'], params.get("bq"))
    K = linear_projection(emb, params['Wk'], params.get("bk"))
    V = linear_projection(emb, params['Wv'], params.get("bv"))
    
    seq_len = len(token_ids)
    cache = init_kv_cache(params["max_seq_len"], K.shape[-1])
    append_kv(cache, K, V)

    atten_out = causal_attention(Q, K, V, is_causal=True)
    hidden = linear_projection(atten_out, params['Wo'], params.get('bo'))

    last_hidden = hidden[-1]
    logits = linear_projection(last_hidden, params["W_out"], params.get("b_out"))

    return logits, cache

# Step 16 - model_decode_step
def model_decode_step(token_id, cache, params):
    """Advance generation by one token using the existing KV cache."""
    # TODO: advance generation by one token using the existing KV cache and return next-token logits
    x = embed_tokens([token_id], params['embedding'])

    q = linear_projection(x, params['Wq'], params.get('bq'))
    k = linear_projection(x, params['Wk'], params.get('bk'))
    v = linear_projection(x, params['Wv'], params.get('bv'))

    append_kv(cache, k, v)
    K = cache['K'][:cache['length']]
    V = cache['V'][:cache['length']]

    attn_out = causal_attention(q, K, V, is_causal = False)[0]
    hidden = linear_projection(attn_out, params['Wo'], params.get('bo'))

    logits = linear_projection(hidden, params['W_out'], params.get('b_out'))

    return np.squeeze(logits), cache

# Step 17 - blocks_needed
def blocks_needed(num_tokens, block_size):
    # TODO: return the number of fixed-size blocks needed to store num_tokens tokens.
    if num_tokens == 0:
        return 0

    return (num_tokens + block_size - 1) // block_size

# Step 18 - init_block_allocator
def init_block_allocator(num_blocks, block_size, d_model):
    # TODO: build the paged KV allocator dict with K_blocks, V_blocks, free_list, seq_tables, and config.
    
    block_allocator = {}
    block_allocator['num_blocks'] = num_blocks
    block_allocator['block_size'] = block_size
    block_allocator['d_model'] = d_model

    block_allocator['K_blocks'] = np.zeros((num_blocks, block_size, d_model), dtype = np.float32)
    block_allocator['V_blocks'] = np.zeros((num_blocks, block_size, d_model), dtype = np.float32)

    block_allocator['free_list'] = list(range(num_blocks))
    block_allocator['seq_tables'] = {}
    block_allocator['seq_lengths'] = {}

    return block_allocator

# Step 19 - allocate_block
def allocate_block(allocator, seq_id):
    # TODO: pop one free block id and append it to allocator['seq_tables'][seq_id]; raise RuntimeError if OOM.
    if len(allocator["free_list"]) == 0:
        raise RuntimeError("Out of memory: no free KV blocks remaining")
    
    block_id = allocator['free_list'].pop()

    if seq_id not in allocator["seq_tables"]:
        allocator["seq_tables"][seq_id] = []

    allocator["seq_tables"][seq_id].append(block_id) # mark for a seq which block belongs to it
    
    return block_id

# Step 20 - free_block
def free_block(allocator, block_id):
    # TODO: return block_id to allocator['free_list']
    
    allocator['free_list'].append(block_id)

# Step 21 - append_to_paged_cache
def append_to_paged_cache(allocator, seq_id, k_new, v_new):
    """Write t new K/V rows into the sequence's paged blocks, allocating as needed."""
    # TODO: append k_new/v_new (t, d_model) rows into seq_id's paged blocks, growing the block table when needed.
    
    assert k_new.shape == v_new.shape, "k_new and v_new must have the same shape"

    t, d_model = k_new.shape
    block_size = allocator["block_size"]

    # Initialize per-sequence metadata if this is a new sequence
    if seq_id not in allocator["seq_tables"]:
        allocator["seq_tables"][seq_id] = []

    if seq_id not in allocator["seq_lengths"]:
        allocator["seq_lengths"][seq_id] = 0

    # Start appending from the current sequence length
    pos = allocator["seq_lengths"][seq_id]
    written = 0

    while written < t:
        block_index = pos // block_size
        offset = pos % block_size

        # Allocate a new physical block if needed
        if block_index == len(allocator["seq_tables"][seq_id]):
            block_id = allocate_block(allocator, seq_id)
        else:
            block_id = allocator["seq_tables"][seq_id][block_index]

        space = block_size - offset
        n = min(space, t - written)

        allocator["K_blocks"][block_id, offset:offset + n, :] = k_new[written:written + n, :]
        allocator["V_blocks"][block_id, offset:offset + n, :] = v_new[written:written + n, :]

        written += n
        pos += n

    # Update sequence length after appending
    allocator["seq_lengths"][seq_id] = pos

# Step 22 - gather_kv_from_blocks
def gather_kv_from_blocks(allocator, seq_id):
    # TODO: reconstruct contiguous (length, d_model) K and V from the sequence's paged blocks.
    
    block_ids = allocator['seq_tables'][seq_id]
    seq_lengths = allocator['seq_lengths'][seq_id]
    block_size = allocator['block_size']

    K = allocator['K_blocks'][block_ids]
    V = allocator['V_blocks'][block_ids]

    K = K.reshape(-1, K.shape[-1])[:seq_lengths]
    V = V.reshape(-1, V.shape[-1])[:seq_lengths]

    return K, V

# Step 23 - paged_attention_step
def paged_attention_step(q, allocator, seq_id):
    # TODO: gather K, V for seq_id from the paged allocator and run causal attention with q
    k, v = gather_kv_from_blocks(allocator, seq_id)
    out = causal_attention(q, k, v)

    return out

# Step 24 - free_sequence_blocks
def free_sequence_blocks(allocator, seq_id):
    # TODO: release all blocks owned by seq_id and remove its entry from seq_tables.

    block_ids = allocator['seq_tables'].pop(seq_id, [])
    for block_id in block_ids:
        free_block(allocator, block_id)
    
    # allocator['seq_lengths'].pop(seq_id, None)
    allocator['seq_tables'].pop(seq_id, None)

# Step 25 - kv_blocks_in_use
def kv_blocks_in_use(allocator):
    # TODO: report allocator usage as {'used': int, 'free': int, 'total': int}.
    
    total = allocator['num_blocks']
    used = sum([len(allocator['seq_tables'][seq_id]) for seq_id in allocator['seq_tables']])
    free = total - used

    return {'used': used, 'free': free, 'total':total}

# Step 26 - make_request
def make_request(request_id, prompt_token_ids, max_new_tokens, sampling_params):
    # TODO: package the request id, prompt tokens, generation budget, and sampling params into a dict.
    request = {}
    request['request_id'] = request_id
    request['prompt_token_ids'] = prompt_token_ids.copy()
    request['max_new_tokens'] = max_new_tokens
    request['sampling_params'] = sampling_params

    return request

# Step 27 - init_sequence_state
def init_sequence_state(request, params):
    # TODO: Initialize per-sequence state by running prefill and storing cache/logits.
    
    token_ids = request['prompt_token_ids']
    if isinstance(token_ids, int):
        token_ids = [token_ids]

    logits, cache = model_prefill(token_ids, params)

    seq_state = {}
    seq_state['request_id'] = request['request_id']
    seq_state['prompt_token_ids'] = token_ids.copy()
    seq_state['generated'] = []

    seq_state['max_new_tokens'] = request['max_new_tokens']
    seq_state['sampling_params'] = request['sampling_params']

    seq_state['last_logits'] = logits
    seq_state['cache'] = cache

    seq_state['done'] = False

    return seq_state

# Step 28 - sequence_decode_step
def sequence_decode_step(state, params, rng):
    # TODO: sample next token from state['last_logits'], advance cache via model_decode_step, append token.
    # sampling and advance

    # sampling
    logits = state['last_logits']
    sp = state['sampling_params']
    greedy = sp.get('greedy', None)
    temperature = sp.get('temperature', None)
    top_k = sp.get('top_k', None)
    top_p  = sp.get('top_p ', None)

    if (greedy is not None and greedy) or (temperature is not None and temperature <= 0):
        next_id = greedy_select(logits)
    else:
        if top_k is not None:
            logits = top_k_filter(logits, top_k)
        if top_p is not None:
            logits = top_p_filter(logits, top_p)
        if temperature is not None:
            logits = apply_temperature(logits, temperature)
        probs = stable_softmax(logits)

        next_id = sample_from_probs(probs, rng)
    
    next_id = int(next_id)

    # advance
    # add to cache

    new_logits, cache = model_decode_step(next_id, state['cache'], params)
    state['cache'] = cache
    state['last_logits'] = new_logits
    state['generated'].append(next_id)

    return next_id, state

# Step 29 - is_sequence_done
def is_sequence_done(state, eos_token_id):
    # TODO: return True if state has hit max_new_tokens budget or last generated token is EOS
    # edge case
    if len(state['generated']) == 0:
        return False
    if state['max_new_tokens'] == 0 or len(state['generated']) == state['max_new_tokens']:
        return True

    last_token_id = state['generated'][-1]

    return last_token_id == eos_token_id

# Step 30 - generate_single_sequence
def generate_single_sequence(request, params, eos_token_id, rng):
    # TODO: drive end-to-end generation for one request and return only the generated token ids.
    
    # init seq state
    state = init_sequence_state(request, params)

    # while not is_seq_done
    # seq_decode_step
    while not is_sequence_done(state, eos_token_id):
        next_id, state = sequence_decode_step(state, params, rng)
    
    return state['generated']

# Step 31 - build_batch_step_input
import numpy as np

def build_batch_step_input(sequences):
    # TODO: collect the last token id from each non-done sequence into a (B,) int64 array.
    
    input_ids = []
    active_indices = []

    for i, seq in enumerate(sequences):
        if not seq['done']:
            active_indices.append(i)
            input_ids.append(seq['token_ids'][-1])
        
    return {'active_indices': active_indices, 
            'input_ids': np.array(input_ids, dtype=np.int64)}

# Step 32 - batched_decode_step
def batched_decode_step(params, sequences, sampling_config):
    """Run one synchronized decode step across active sequences."""
    # TODO: For each active sequence, run a decode step and append the sampled token.
    
    # build batch input
    batch  = build_batch_step_input(sequences)
    greedy = sampling_config['greedy']

    # decode
    for idx, tok in zip(batch['active_indices'], batch['input_ids']):
        state = sequences[idx]
        logits, cache = model_decode_step(int(tok), state['kv_cache'], params)

        if greedy is None or greedy:
            next_id = greedy_select(logits)
        else:
            if 'top_k' in sampling_config:
                logits = top_k_filter(logits, sampling_config['top_k'])
            if 'top_p' in sampling_config:
                logits = top_p_filter(logits, sampling_config['top_p'])
            if 'temperature' in sampling_config:
                logits = apply_temperature(logits, sampling_config['temperature'])
            probs = stable_softmax(logits)

            next_id = sample_from_probs(probs, sampling_config['rng'])

        state['token_ids'].append(int(next_id))
  
    return sequences

# Step 33 - static_batch_generate
def static_batch_generate(params, requests, sampling_config, max_new_tokens):
    """Run prefill for all requests, then iterate batched decode steps until each
    sequence hits its per-request budget or the global max_new_tokens cap."""
    # TODO: prefill each request, then loop sampling next tokens until done.
    greedy = sampling_config.get('greedy', None)
    out = []
    states = [init_sequence_state(request, params) for request in requests]
    greedy = sampling_config['greedy']

    # decode
    for _ in range(max_new_tokens):
        all_done = True

        for state in states:
            cap = min(state['max_new_tokens'], max_new_tokens)
            if state['done'] or len(state['generated']) >= cap:
                state['done'] = True
                continue
            all_done = False

            logits = state['last_logits']

            if greedy is None or greedy:
                next_id = greedy_select(logits)
            else:
                if 'top_k' in sampling_config:
                    logits = top_k_filter(logits, sampling_config['top_k'])
                if 'top_p' in sampling_config:
                    logits = top_p_filter(logits, sampling_config['top_p'])
                if 'temperature' in sampling_config:
                    logits = apply_temperature(logits, sampling_config['temperature'])
                probs = stable_softmax(logits)

                next_id = sample_from_probs(probs, sampling_config['rng'])

            next_id = int(next_id)

            new_logits, cache = model_decode_step(next_id, state['cache'], params)
            state['cache'] = cache
            state['last_logits'] = new_logits
            state['generated'].append(next_id)
            if len(state['generated']) >= cap:
                state['done'] = True

        if all_done:
            break
        
    return [{'request_id': state['request_id'], 'output_ids': state['generated']} for state in states]

# Step 34 - has_free_capacity
def has_free_capacity(allocator, required_blocks):
    # TODO: return True iff allocator has at least required_blocks free blocks.
    
    return required_blocks <= len(allocator['free_list'])

# Step 35 - continuous_batch_step
def continuous_batch_step(params, running, allocator, sampling_config):
    """Advance every active sequence in `running` by one decoded token using the paged allocator."""
    # TODO: for each non-done sequence, project Q/K/V from its last token, append to the paged cache, run paged attention, sample, and append.

    emb = params['embedding']
    Wq = params['Wq']
    Wk = params['Wk']
    Wv = params['Wv']
    W_out = params['W_out']
    d_model = emb.shape[-1]
    greedy = sampling_config.get('greedy', False)
    eos_token_id = sampling_config.get('eos_token_id', None)

    for state in running:
        if state['done']:
            continue

        seq_id = state['request_id']
        token_id = state['token_ids'][-1]

        # x is 1D: (d_model,)
        x = emb[token_id]

        q = x @ Wq
        k = x @ Wk
        v = x @ Wv


        q = np.asarray(q).reshape(1, -1)

        # k/v must be 2D because append_to_paged_cache expects (t, d_model)
        k = np.asarray(k).reshape(1, -1)
        v = np.asarray(v).reshape(1, -1)

        append_to_paged_cache(allocator, seq_id, k, v)

        attn_out = paged_attention_step(q, allocator, seq_id)

        logits = linear_projection(attn_out, W_out)

        if hasattr(logits, "ndim") and logits.ndim == 2:
            logits = logits[-1]

        if greedy:
            next_id = greedy_select(logits)
        else:
            temperature = sampling_config.get('temperature', 1.0)
            top_k = sampling_config.get('top_k', None)
            top_p = sampling_config.get('top_p', None)

            if temperature is not None:
                logits = apply_temperature(logits, temperature)

            if top_k is not None:
                logits = top_k_filter(logits, top_k)

            if top_p is not None:
                logits = top_p_filter(logits, top_p)

            probs = stable_softmax(logits)
            next_id = sample_from_probs(probs, sampling_config['rng'])

        next_id = int(next_id)

        state['token_ids'].append(next_id)
        state['generated'].append(next_id)
        state['length'] += 1

        if len(state['generated']) >= state['max_new_tokens']:
            state['done'] = True

        if eos_token_id is not None and next_id == eos_token_id:
            state['done'] = True

    return running

# Step 36 - run_continuous_batching
import math
def run_continuous_batching(params, requests, allocator, sampling_config, max_steps):
    # TODO: Drive the continuous-batching loop: admit, decode, retire finished sequences.
    
    waiting = requests
    running = []
    completed = []
    block_size = allocator['block_size']
    eos_token_id = sampling_config['eos_token_id']

    step = 0
    while len(waiting)+len(running) > 0 and step < max_steps:
        
        # add request from waiting to running (NEED TO GET BETTER LOGIC)
        while len(waiting) > 0:
            request = waiting[0]

            prompt_len = len(request['prompt_token_ids'])
            max_seq_len = prompt_len + request['max_new_tokens']
            required_blocks = math.ceil(max_seq_len / block_size)

            if has_free_capacity(allocator, required_blocks):

                state = {
                            'request_id': request['request_id'],
                            'token_ids': request['prompt_token_ids'],
                            'generated': [],
                            'length': len(request['prompt_token_ids']),
                            'max_new_tokens': request['max_new_tokens'],
                            'done': False
                        }
                running.append(state)
                waiting.pop(0)
            else:
                break

        # process current running
        if len(running) == 0:
            break
        running = continuous_batch_step(params, running, allocator, sampling_config)

        # retire states
        done_mask = [is_sequence_done(state, eos_token_id) for state in running]
        if sum(done_mask) > 0:
            finished = [state for state, done in zip(running, done_mask) if done]
            running = [state for state, done in zip(running, done_mask) if not done]

            for state in finished:
                seq_id = state['request_id']
                free_sequence_blocks(allocator, seq_id)

                completed.append({'request_id': seq_id, 'output_ids': state['generated']})
        
        step += 1

    # If max_steps stopped the loop, return partial outputs too.
    for state in running:
        seq_id = state['request_id']

        free_sequence_blocks(allocator, seq_id)

        completed.append({
            'request_id': seq_id,
            'output_ids': state['generated']
        })
    
    return completed

# Step 37 - priority_queue_push
import heapq

def priority_queue_push(heap, priority, request):
    # TODO: push (priority, counter, request) onto the min-heap with stable tie-breaking
    counter = len(heap)
    heapq.heappush(heap, (priority, counter, request))
    return heap

# Step 38 - priority_queue_pop
import heapq

def priority_queue_pop(heap):
    # TODO: Pop and return the request with the smallest priority from the min-heap, or None if empty.
    if len(heap) == 0:
        return None
    else:
        return heapq.heappop(heap)[2]

# Step 39 - select_admissions
def select_admissions(waiting_heap, allocator, block_size, max_admit):
    # TODO: Pop requests from the waiting priority queue and admit as many as the allocator can host.
    count = 0
    admitted = []
    required_blocks = 0

    while len(waiting_heap) > 0 and count < max_admit:
        # check the top heap request capacity
        request = waiting_heap[0][2]

        prompt_len = len(request['prompt_token_ids'])
        required_blocks += math.ceil(prompt_len / block_size)

        if has_free_capacity(allocator, required_blocks):
            request = priority_queue_pop(waiting_heap)

            admitted.append(request)
            count += 1
        else:
            break
    
    return admitted

# Step 40 - preempt_sequence
def preempt_sequence(sequence, allocator, waiting_heap):
    # TODO: free the sequence's KV blocks and re-queue its request on the waiting heap.
    
    seq_id = sequence['request_id']
    priority = sequence['priority']

    # free cache
    free_sequence_blocks(allocator, seq_id)

    # re-enqueue waiting_heap
    # recreate request record to remove current runtime info
    request = {
        'request_id': seq_id,
        'max_new_tokens': sequence['max_new_tokens'],
        'prompt_token_ids': sequence['prompt_token_ids'],
        'priority': priority,
    }

    priority_queue_push(waiting_heap, priority, request)
    
    return request

# Step 41 - schedule_step
def schedule_step(waiting_heap, running, allocator, block_size, max_running):
    # TODO: preempt over-capacity sequences, then admit from the waiting heap up to max_running.

    while len(running) > max_running:
        sequence = running.pop()
        request = preempt_sequence(sequence, allocator, waiting_heap)

    admitted = select_admissions(waiting_heap, allocator, block_size, max_running - len(running))

    # schedule admission but not yet modify running
    out = {
        'running': running,
        'newly_admitted': admitted,
    }

    return out

# Step 42 - format_stream_chunk (not yet solved)
# TODO: implement

# Step 43 - submit_request (not yet solved)
# TODO: implement

# Step 44 - drive_until_complete (not yet solved)
# TODO: implement

# Step 45 - collect_request_output (not yet solved)
# TODO: implement

# Step 46 - build_completion_response (not yet solved)
# TODO: implement

# Step 47 - time_to_first_token (not yet solved)
# TODO: implement

# Step 48 - inter_token_latency (not yet solved)
# TODO: implement

# Step 49 - aggregate_throughput (not yet solved)
# TODO: implement

# Step 50 - latency_percentiles (not yet solved)
# TODO: implement

# Step 51 - run_throughput_latency_benchmark (not yet solved)
# TODO: implement

