## Performance Analysis for General, Multiplicatice,Additive Attention

| Attentions   |  Average Time per epoch|Inference Time   | Model Size (MB)|  Test Loss   |  Test PPL   |
|--------|------------------|--------|------------------|--------|--------|
| General Attention | 117.003         | 0.0169 | 49.35       | 5.101  | 164.140  | 
|  Multiplicative Attention| 126.415        | 0.0049 | 49.35        |5.051  | 156.174  | 
| Additive Attention | 158.454   | 0.0049 | 49.35         | 5.031  | 153.126  | 

From the above analysis and evalution we can see that average time per epoch for general, multiplicative and additive attention are 117.003s, 126.415s, 158.454s respectively, where as model size for all three model is same i.e. 49
35 MB.

<br>
<br>

## Graph Report
### General Attention 
![](https://github.com/ishika28/NLP_Assignment/blob/main/A3/Screenshot/general.png)
<br>

### Multiplicative Attention
![](https://github.com/ishika28/NLP_Assignment/blob/main/A3/Screenshot/Multiplicative.png)
<br>

### Additive Attention
![](https://github.com/ishika28/NLP_Assignment/blob/main/A3/Screenshot/additive.png)

<br>

To run the file local for A3
1. First go to the A3 folder
2. Run the A3.ipynb file
3. Then you will gets the model
4. To run app.py file. In terminal write python App/app.py
5. Application will be open at 127.0.0.1:8050

## Demo
![](https://github.com/ishika28/NLP_Assignment/blob/main/A3/demo_A3.gif)
