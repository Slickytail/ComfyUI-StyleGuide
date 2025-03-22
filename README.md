# ComfyUI StyleGuide

StyleGuide - ComfyUI Replication.

Implements the algorithm from the paper [StyleGuide: Preventing Content Leakage using Negative Query Guidance.](https://openreview.net/forum?id=618qfjvSt9)

The implementation is in the early stages, and may be incompatible with more complex workflows.  
The main weaknesses we observe at this moment is that increasing the CFG easily "overcooks" the style,
and in cases where a high CFG is required for correct anatomy generation, the tradeoff may be limiting.  
Future work on the implementation will likely support a 3 or 4-pass computation with multiple guidance values (ie, uncond, cond, visual query, negative visual query).  
We also find that the output images can be a little bit blurry -- this is likely to be solved by turning off the style injection during the last few generation timesteps.  
Additionally, the algorithm has trouble replicating photographic styles (it works better with animated, painterly, and artistic ones).

Recommended settings:
 - **Use an SDXL finetune, rather than the base model**. In the base model, the subject easily becomes unrecognizable. I've had good results with [RealvisXL5](https://civitai.com/models/139562?modelVersionId=789646), for example.
 - CFG around 3.0
 - A short, not too detailed positive prompt
 - 30 steps
 - Color calibration between 50% and 80% of the timesteps.
 - Start from 25th output layer. Increase this value if the style is too strong or the subject is not clear.
