# Multi-Agent Orchestration for Creative Poster Generation

A modular, agent-based system for generating creative posters using LangGraph orchestration. This project demonstrates how breaking down complex tasks into specialized agents can produce superior results compared to monolithic approaches.

## Architecture

![System Architecture](Images/Pasted%20image%2020251122132926.png)

The system uses a graph-based multi-agent architecture where each node specializes in a specific task: text generation, image creation, validation, and composition. Agents communicate through a shared state, enabling iterative refinement and quality control.

## Problem Statement

Direct prompting of image generation models produces inconsistent results with common issues:
- Generic, uncreative outputs
- Poor text rendering (spelling errors, blurring)
- Failure to incorporate brand elements (logos)
- Lack of adherence to prompts

**Example of direct prompting failures:**

<div align="center">
<img src="Images/Pasted%20image%2020251122151235.png" width="30%" />
<img src="Images/Pasted%20image%2020251122151405.png" width="30%" />
<img src="Images/Pasted%20image%2020251122151422.png" width="30%" />
</div>

*Direct prompting results: lacking creativity, text rendering issues, generic outputs*

## Solution: Modular Multi-Agent Pipeline

### Key Design Principles
- **Modularity**: Each agent handles a specific subtask, making debugging and improvements tractable
- **Iterative Refinement**: Validation agents provide feedback for quality control
- **State Management**: LangGraph maintains consistency across agent interactions
- **Specialization**: Simple models orchestrated effectively can outperform complex monolithic approaches

### Technical Improvements Implemented

1. **Dual-Stage Image Generation**: Separated background generation from text overlay to improve quality
2. **Classifier-Free Guidance (CFG) Tuning**: Optimized the trade-off between prompt adherence and diversity
   - Formula: $\hat{\epsilon}_{\theta}((x_{t}), c(s), t) = (1+w)\epsilon_{\theta} (x_{t}, c(s), t) - w \epsilon_{\theta}(x_{t}, c(\theta), t)$
3. **Negative Prompting**: Explicitly guide model away from undesired outputs
   - Revised formula: $\hat{\epsilon}_{\theta}((x_{t}), c(s), t) = (1+w)\epsilon_{\theta} (x_{t}, c(p_{+}), t) - w \epsilon_{\theta}(x_{t}, c(p_{-}), t)$
4. **Multi-Stage Validation**: Separate validation for text correctness and rendering quality
5. **Increased Denoising Steps**: Better quality at the cost of generation time

## Results

### Input
- **Prompt**: "Italian Restaurant"
- **Company Logo**: Provided brand asset

### Output Progression

**Early Pipeline Results:**

![Early Result](Images/Pasted%20image%2020251122131317.png)

**After Optimizations:**

<div align="center">
<img src="Images/Pasted%20image%2020251122130615.png" width="45%" />
<img src="Images/Pasted%20image%2020251122130630.png" width="45%" />
</div>

<div align="center">
<img src="Images/Pasted%20image%2020251122131022.png" width="45%" />
<img src="Images/poster.png" width="45%" />
</div>

### Generalization Test: Chinese Restaurant

**Input Prompt**: "Chinese Restaurant"

![Chinese Restaurant Input](Images/Pasted%20image%2020251122181646.png)

**Generated Output**:

![Chinese Restaurant Result](Images/Pasted%20image%2020251122182047.png)

## Technology Stack

- **Orchestration**: LangGraph (open-source framework for stateful multi-agent systems)
- **Image Generation**: Diffusion models with CFG and negative prompting
- **Image Editing**: Qwen-Image-Edit for text overlay
- **Validation**: LLM-based quality assessment
- **Infrastructure**: GPU-accelerated inference pipeline

## Why This Approach Works

1. **Modular Design**: Easy to fix errors, swap models, or adapt to new tasks
2. **Performance**: Quality-time trade-offs can be optimized per component
3. **Scalability**: New capabilities can be added as additional agents
4. **Transparency**: Each step is inspectable and debuggable
5. **Generalizability**: Unlike specialized fine-tuned models, this adapts to various poster types

## Key Learnings

1. **Agent Orchestration > Monolithic Models**: Breaking complex tasks into specialized agents produces better results than single-shot generation
2. **Validation is Critical**: Separate validation agents with specific criteria dramatically improve output quality
3. **Parameter Tuning Matters**: CFG scale, denoising steps, and negative prompts significantly impact results
4. **Model-Specific Adaptation**: Instructions can be fine-tuned to work around specific model limitations (e.g., Qwen-Image-Edit's spelling issues)

## Challenges Overcome

- Limited GPU resources for image generation and editing
- Model reliability issues (Qwen-Image-Edit spelling, Grok 4.1-fast sarcastic feedback)
- Balancing generation speed vs. quality trade-offs
- Designing effective validation criteria for creative outputs

## Evaluation Methodology

Image generation lacks universal benchmarks for niche tasks like piecewise poster creation. We focused on:

- **Creativity & Quality**: Visual appeal and originality
- **Brand Integration**: Proper logo incorporation
- **Text Quality**: Minimal spelling errors and blurring
- **Pipeline Behavior**: Modular components working as intended

Human evaluation remains the gold standard for such creative tasks, supplemented by targeted automated checks.

---

**Project Type**: Academic Research | Multi-Agent Systems | Creative AI\
**Technologies**: LangGraph, Diffusion Models, LLM Validation, Python
