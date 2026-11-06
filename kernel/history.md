# How we got here: the Qwen phase

Before Nemotron, the project compiled a stack machine into Qwen3.5-0.8B. That work lives in an earlier research repository. This page keeps its results and its lesson.

The idea came from [qwenputer](https://github.com/divagr18/qwenputer) by Divyansh Agrawal, which compiles a RISC-V calculator into a language model's weights. The Qwen phase reused its MIT-licensed weight compiler and its carry-lookahead construction. The Nemotron kernel in this repository is a new construction for Mamba-2 layers and does not use that code.

## What worked

| Result | Number |
|---|---|
| A Forth circuit vs qwenputer's RISC-V circuit on the same sparse executor, 23 paired runs | 3.90 times faster machine execution, every answer and every predicted machine token right |
| A persistent stack and 16 variables in extra Gated DeltaNet layers, through emitted weights on MLX | 10,000 no-op updates with the memory exactly preserved |
| A compiled decimal readout of signed 64-bit results | 16/16, 256/256 and 185/185 on exhaustive and boundary tests |

The machine was exact. Execution was never the problem.

## What did not work

The 0.8B model could not drive the machine. With a fixed, human-written schedule of actions, the machine was right. With actions chosen by the model, it failed:

| Experiment | Result |
|---|---|
| Model-selected actions on 17 math questions | 5 of 17 correct machine reads |
| A small controller LoRA | Failed its development gate, not kept |
| Controller RL pilot | 10/16 to 11/16 on development tasks (one formatting fix), not promoted |
| A 20-action autonomous task | Not reached: the model repeated its plan and never accepted an action |

## The lesson

An exact machine needs a model that can plan. So we moved to a strong reasoner, Nemotron 3.5 Lightning, and later to plain-text Forth that the model already knows. The Mamba-2 layers of Nemotron have no delta-rule overwrite, so we designed the kernel again from the start ([the kernel](README.md#numbers-in-a-mamba-2-state)).
