# UnslothAI Hiring Challenge: Memory Efficient Backprop

## Overview
This repository is part of the **UnslothAI Hiring Challenge**, inspired by a Twitter challenge posted by **Daniel Han (@danielhanchen)**. The challenge consists of five technical tasks, each designed to test and push the boundaries of AI engineering and optimization skills.

Tweet Link: [https://x.com/danielhanchen/status/1891194528931209644](https://x.com/danielhanchen/status/1891194528931209644)

I will be updating this repository as I complete each task. Currently, I have implemented **Task E: Memory Efficient Backprop**, optimizing backpropagation to reduce VRAM usage while maintaining accuracy.

---

## Task E: Memory Efficient Backprop
### Problem Statement
In large language models (LLMs), the final projection matrix is used to compute token probabilities, i.e., \( \sigma(XW) \). However, when dealing with a large vocabulary size (e.g., 128K), storing logits in memory causes VRAM spikes. For instance:

- **Batch Size (bsz)** = 4
- **Sequence Length (qlen)** = 4096
- **Hidden Dimension (hd)** = 4096
- **Vocabulary Size (vocab)** = 128K

This results in a memory requirement of **4GB (bfloat16)** and potentially **8GB (float32 upcast)**.

### Solution
To optimize memory efficiency, we leverage chunking during forward and backward passes, avoiding the full materialization of logits in memory. This is achieved using **custom autograd functions** in PyTorch.

### Implementation Highlights
- Uses **chunked processing** to split tensors into smaller segments.
- Prevents unnecessary upcasting to float32.
- Preserves computational correctness while reducing VRAM usage.

### Performance Comparison
| Method          | Time (sec) | VRAM Usage (GiB) |
|----------------|-----------|------------------|
| Standard       | 3.195809  | 8.014037        |
| Custom (Ours)  | 4.204161  | 3.362484        |

The implementation achieves a significant **VRAM reduction (~58%)**, making it an efficient approach for large-scale LLM training.

---

## Next Steps
I will continue solving the remaining tasks in this challenge and update this repository with further implementations and optimizations.

Stay tuned for more updates!

---

## References
- [UnslothAI Hiring Challenge Tweet](https://x.com/danielhanchen/status/1891194528931209644)
- [Kaggle Notebook (Current Implementation)](https://www.kaggle.com/code/yash9439/unsloth-taske)

## Connect
Feel free to reach out or collaborate:
- **LinkedIn**: [linkedin.com/in/yash-bhaskar](https://www.linkedin.com/in/yash-bhaskar)
- **GitHub**: [github.com/Yash9439](https://github.com/Yash9439)
- **Medium**: [medium.com/@yash-bhaskar](https://medium.com/@yash-bhaskar)

