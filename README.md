# Alpaca Lora 4bit, long-range and other experiments

Forked from: https://github.com/johnsmith0031/alpaca_lora_4bit.git

LLaMA is pretrained using rotary embeddings spanning 2048 tokens. As a result it becomes useless for generation on longer sequences than that.

With quantized weights and memory-efficient attention it's feasible to run inference on much longer sequences, though, even on low-end hardware. This project is to test if a low-rank adapter can make up for LLaMA's lack of training on longer sequences.

## Results

Success, I think?

It took about 5 hours to train for 3 epochs on one A100. Total VRAM usage was 75 GB throughout training. Settings:

- Using memory-efficient attention from xformers
- Position embedding table extended to 6144 tokens
- Training data was around 3,200 random items from C4 of at least 6142 tokens in length, truncated to 6142 tokens
- LoRA rank of 32 (not sure how much this matters)
- Base model is LLaMA-13B quantized to 4 bits, group size of 128

To set a baseline, I first trained on shorter sequences (up to 2048 tokens) from the same dataset. The resulting loss ended up around 2.0 in all cases, if not a little higher for very short sequences. Convergence was very quick, taking only a few iterations. I'm assuming this makes sense since the LoRA is essentially just training to become a no-op in this case, as LLaMA is presumably quite well rehearsed on C4 already.

The loss can drop substantially below 2.0, but apparently not with this training data. Lower loss seems to require some level of redundancy in the training examples, like from repeating sequences, or a question paired with an answer that the base model can derive from the question.

With a sequence length of 6144, the adapted model starts on a much higher loss (5.0+), which was expected. It does however converge to about 2.0 after a few hundred iterations, which is encouraging.

This doesn't mean that it's actually performing attention across the whole length of the sequence (need more tests to find out), but at least it isn't hopelessly confused by longer sequences like the base model is.    

Generating with the adapted model does produce coherent text, up to 6144 tokens. After that it quickly degenerates, exactly as happens with the base model around the 2048-token mark. So this is a good sign, too.

## Notes on inference

Inference on the 13B model with a sequence length of 6144 takes a fixed 7.1 GB of VRAM (on top of the base model and LoRA weights).

To achieve this I had to use a modified version of the xformers attention patch for inference, shown in llama_attn_hijack_xformers_modified.py for refence. The code I added is ugly, largely written by GPT-4, and I'm honestly a little surprised it works. But there it is anyway. For reference.

The only change is to pre-allocate tensors for the key/value cache with enough space for the full sequence length, and then to copy new data in-place into the same underlying storage instead of concatenating with every new token.

While it shouldn't make a big difference in theory, I'm pretty sure that the repeated concatenations made for a very unfortunate pattern of memory allocations, causing bad fragmentation and lots of wasted VRAM. Enabling the cache ultimately used up several times the amount of memory actually needed for the cache itself.

The pre-allocated cache needs less copying of tensor data, but on the other hand cache locality is worse, so I'm not sure if it's faster or slower this way.  

## Other notes

Had to disable the monkey patch from the last commit before forking from the main repo as it breaks saving the adapter after training.

# TODO: 

- Need to test how well the adapted model attends across the whole sequence length. There's not much point if the LoRA is simply learning to ignore tokens outside the range that the base model was trained on.
- Need to experiment with other training data, perhaps something deliberately crafted to encourage long-range attention. 
- I'm aiming for this to run on consumer GPUs (up to 24 GB), and even with the improved attention, 6144 tokens still requires a total of 16.7 GB including the model and LoRA. I'd like to go even longer, and ideally I'd want to run long sequences on the quantized 30B model, too. Calls for some more work on the modified attention function.
- Lots of hyperparameters to randomly fiddle with.
- Want to finetune for tasks like summarization, chat, etc., but with longer attention span.