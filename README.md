<div align="center">

# Tibeb

**ጥበብ · "wisdom" in Amharic**

An open large language model project built on DeepSeek open weights

[![Weights on Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Weights-rekidane%2FTibeb--Pro-ffc107)](https://huggingface.co/rekidane/Tibeb-Pro)
[![License: MIT](https://img.shields.io/badge/License-MIT-f5de53)](LICENSE-CODE)

</div>

## About

Tibeb is maintained by [Kidanemaryam W. Reta](https://rekidane.github.io/kidane/). This repository holds Tibeb's code and documentation; the model weights are hosted on Hugging Face.

| Model | Base | Weights |
| :--- | :--- | :--- |
| Tibeb-Pro v0.1 | DeepSeek-V4-Pro-0813 (about 1.6T parameters, 49B active, 1M-token context) | [rekidane/Tibeb-Pro](https://huggingface.co/rekidane/Tibeb-Pro) |

Tibeb-Pro v0.1 uses the unchanged DeepSeek-V4-Pro-0813 weights with a default Tibeb identity added to the chat encoder. Its capabilities and benchmark results are those of the base model until Tibeb-specific training is released.

## Roadmap

1. **Amharic proficiency:** continued training and instruction tuning on Amharic grammar, dictionary and text resources, evaluated on Amharic benchmarks before release.
2. **Identity tuning:** fine-tuning so the model identifies as Tibeb without a system prompt.
3. **Smaller variants:** a lighter Tibeb model for users without multi-GPU servers.

## Repository contents

| Folder | Contents |
| :--- | :--- |
| [`encoding/`](encoding) | DeepSeek-V4 chat encoder with the default Tibeb system prompt, plus its test suite (`python encoding/test_encoding_dsv4.py`) |
| [`inference/`](inference) | Legacy DeepSeek-V3 reference inference code from the original fork; it does not run V4 weights. Use the `inference/` folder in the [Hugging Face repo](https://huggingface.co/rekidane/Tibeb-Pro/tree/main/inference) for Tibeb-Pro |

## Quick start

```python
from encoding_dsv4 import encode_messages

messages = [{"role": "user", "content": "Who are you?"}]
prompt = encode_messages(messages, thinking_mode="chat")
# Tibeb system prompt is prepended when no system message is given
```

Serving instructions for vLLM and SGLang are in the [Tibeb-Pro model card](https://huggingface.co/rekidane/Tibeb-Pro).

## License and attribution

Tibeb's code and the Tibeb-Pro weights are released under the MIT License. They are derived from DeepSeek-V3 and DeepSeek-V4 by DeepSeek-AI, whose MIT copyright notice is retained in [`LICENSE-CODE`](LICENSE-CODE). [`LICENSE-MODEL`](LICENSE-MODEL) is the original DeepSeek-V3 model agreement, kept with the legacy V3 code. Tibeb is an independent project and is not affiliated with or endorsed by DeepSeek.

## Citation

```
@misc{reta2026tibeb,
      title={Tibeb-Pro: an open model built on DeepSeek-V4},
      author={Reta, Kidanemaryam W.},
      year={2026},
      howpublished={\url{https://huggingface.co/rekidane/Tibeb-Pro}},
}

@misc{deepseekai2026deepseekv4,
      title={DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence},
      author={DeepSeek-AI},
      year={2026},
}
```
