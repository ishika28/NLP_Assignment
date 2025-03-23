## Dataset 

I used dataset from hugging face, name [OxAISH-AL-LLM/wiki_toxic](https://huggingface.co/datasets/OxAISH-AL-LLM/wiki_toxic)
<br>

## Training metrics
Even Layer Model
![](https://github.com/ishika28/NLP_Assignment/blob/main/A7/image/even.png)

<br>

odd Layer Model
![](https://github.com/ishika28/NLP_Assignment/blob/main/A7/image/odd.png)

<br>

##  Evaluation and Analysis
### Even Layer Model

This model uses only the even-numbered layers of BERT (e.g., layers 0, 2, 4, etc.), reducing the model’s size. In term of Performance, it performed both test sentences ("I love you" as non-toxic, "I hate you" as toxic), suggesting better generalization than the odd layers model. The training and validation loss plots look stable, and it likely benefits from the teacher’s guidance via distillation.Even layers might preserve more of BERT’s initial processing (starting at layer 0), giving it a slight edge over odd layers. Still, it’s only half the model, so it’s not as powerful as the full BERT.


## Odd Layer Model

This uses only the odd-numbered layers (e.g., layers 1, 3, 5, etc.), also a slimmed-down version of BERT.Like the even layers model, it’s trained with the same loss combination. In term of performance, the inference test shows it correctly identifies "I hate you" as toxic but mislabels "I love you" as toxic. This suggests it might be overly aggressive in flagging text as toxic, possibly due to overfitting or poor generalization. Without test set accuracy, we can’t say for sure, but the validation loss plot shows a downward trend, indicating it learns something but just not perfectly.Using only odd layers cuts the model’s depth in half, potentially losing critical contextual understanding that BERT relies on across all layers. The loss terms try to align it with the teacher, but the reduced capacity might limit its ability to capture nuanced patterns.

## LoRA Model

This uses the full BERT model but applies LoRA to adapt it efficiently, adding low-rank matrices to the attention layers (query, key, value). In terms of performance, the validation accuracy peaks at 0.9328, which is close to the teacher’s best test accuracy (0.9400). Inference on the test sentences is spot-on, matching the even layers model’s success. Given its validation strength, it’s reasonable to assume test accuracy would be in the 0.92–0.94 range.LoRA keeps all 12 layers but only fine-tunes a tiny fraction of parameters, making it efficient without sacrificing much power. It adapts the full BERT architecture to the task, leveraging its full contextual understanding.
<br>

## Observation

| Model Type   |  Training Loss| Avg Metric |
|--------|------------------|--------|
| Odd Layer| 0.9290         | 1.84120   |  
|  Even Layer|0.9280        | 0.91980  | 



LoRA Model 

|Step	|Training Loss|	Validation Loss	|Accuracy |
|-------|--------------|--------------|------------|
|500	|0.287200	|0.206477	|0.914539|
|1000	|0.187400	|0.295403	|0.885555|
|1500	|0.177000	|0.171205   |0.932790|
|2000	|0.154300	|0.176234	|0.932085|
|2500	|0.149600	|0.221862	|0.917124|

<br>

### Challenges encountered



- Class Imbalance Issue:

  - Training data had a stark 90:10 class imbalance (91,778 non-toxic vs. 10,346 toxic).

  - Models overpredicted the majority class.

  - No direct mitigation but used downsampling.

- Distilled Models (Odd/Even Layers)
   -  Improper balancing could lead to one loss dominating, skewing learning. Without fine-tuning, the model may prioritize mimicking the teacher over learning the task itself.

   - The odd layers model’s tendency to overpredict "toxic" suggests it struggled with imbalance, which overwhelmed its limited capacity. With fewer toxic samples, generalization suffered.

   - Cutting BERT’s 12 layers to 6 significantly weakened its ability to capture complex patterns.Halving depth risks losing nuance, and distillation alone can’t fully compensate if the model is too small.

- LoRA (Low-Rank Adaptation)
    -  Required extra steps to integrate with the peft library and Trainer, including configuring LoraConfig, wrapping the model with get_peft_model, and ensuring compatibility with TrainingArguments.

    - Despite training fewer parameters, LoRA still used the full BERT architecture, leading to higher inference and memory usage than distilled models. 

    - Despite a strong validation accuracy of 0.9328, the lack of downsampling or class weighting meant the model might still be biased.

    -  LoRA trained for 2 epochs, longer per epoch than distilled models due to using the full BERT architecture. While parameter efficiency improved
## Demo
![](https://github.com/ishika28/NLP_Assignment/blob/main/A7/demo_video.gif)