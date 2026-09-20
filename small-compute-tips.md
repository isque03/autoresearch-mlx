Seeing as there seems to be a lot of interest in tinkering with autoresearch on much smaller compute platforms than an H100, a few extra words. If you're going to try running autoresearch on smaller computers (Macbooks etc.), I'd recommend one of the forks below. On top of this, here are some recommendations for how to tune the defaults for much smaller models for aspiring forks:

To get half-decent results I'd use a dataset with a lot less entropy, e.g. this TinyStories dataset. These are GPT-4 generated short stories. Because the data is a lot narrower in scope, you will see reasonable results with a lot smaller models (if you try to sample from them after training).
You might experiment with decreasing vocab_size, e.g. from 8192 down to 4096, 2048, 1024, or even - simply byte-level tokenizer with 256 possibly bytes after utf-8 encoding.
In prepare.py, you'll want to lower MAX_SEQ_LEN a lot, depending on the computer even down to 256 etc. As you lower MAX_SEQ_LEN, you may want to experiment with increasing DEVICE_BATCH_SIZE in train.py slightly to compensate. The number of tokens per fwd/bwd pass is the product of these two.
Also in prepare.py, you'll want to decrease EVAL_TOKENS so that your validation loss is evaluated on a lot less data.
In train.py, the primary single knob that controls model complexity is the DEPTH (default 8, here). A lot of variables are just functions of this, so e.g. lower it down to e.g. 4.
You'll want to most likely use WINDOW_PATTERN of just "L", because "SSSL" uses alternating banded attention pattern that may be very inefficient for you. Try it.
You'll want to lower TOTAL_BATCH_SIZE a lot, but keep it powers of 2, e.g. down to 2**14 (~16K) or so even, hard to tell.
