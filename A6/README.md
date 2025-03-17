## Analysis and Problem Solving

###  List of the retriever and generator models used

1) Retriever Model
<br>
   - Model: FAISS (Fast Approximate Nearest Neighbors Index) with HuggingFaceInstructEmbeddings.
   <br>
Details:
   <br>
   - Embeddings: hkunlp/instructor-base.<br>
   - Vector Store: FAISS, initialized with FAISS.from_documents() and loaded with FAISS.load_local().

2) Generator Model
<br>
- Model: Groq Llama3-70B-8192.<br>

### Analyze any issues related to the models providing unrelated information. 

Building this chatbot was a journey, but I hit some bumps where the models spit out stuff that didn’t quite fit. Here’s what I noticed about the retriever and generator going off-track, plus some ideas to fix it.

- Embedding Model (HuggingFaceInstructEmbeddings)
I used hkunlp/instructor-base to make vectors out of my text, but it’s not perfect. Sometimes it doesn’t really “get” what I’m asking. Like, if I say, “What projects has Ishika done?” it might grab chunks about her school instead of work because the embeddings latch onto vague words like “experience.” It’s a general-purpose model, so it’s not tuned to Ishika’s life story—names like hers or quirky details might throw it off. Also, short questions like “What’s her deal?” can turn into fuzzy vectors that pull random bits.

    - Fixes: Maybe mix in a keyword search with FAISS—like BM25—to catch specific terms better. Or tweak the embeddings with some bio-specific training, though that’s a bit of a project.

- Vector Store (FAISS)
FAISS is awesome for speed, but omly with small files. If I ask something offbeat like “What’s Ishika’s favorite food?” it still picks the “closest” chunks—say, her education—since there’s no food info to find. It’s like forcing a square peg into a round hole. Plus, FAISS goes by vector math, so “Tell me about her job” might grab project details instead of her actual work history if the numbers line up wrong. And with only 4 chunks retrieved, it sometimes misses the good stuff—like details getting buried under noise.
    - Fixes: I could chop the PDF into smaller, overlapping pieces so each chunk has more context. Grabbing more than 4 chunks might help too, or adding a reranking step to sort out what’s actually useful.
- Generator model (Groq Llama3-70B)
The Groq model is a beast at writing, but it’s got a mind of its own. If FAISS hands it weak context—like education chunks for a hobbies question—it might guess “Ishika loves coding” when cv_bio.pdf says nothing about that. It’s also got this huge brain from general training, so it might sneak in random facts. Like, “What’s her job?” could get “She’s a software engineer” even if the bio says something else. I caught it once trying to add “(No document cited...)” before I yanked that out—it’s eager to sound smart, even when it shouldn’t. For “What’s her favorite color?” with no data, it might just pick “Blue” instead of shrugging.
    - Fixes: I’d tweak my prompt to say, “Stick to the docs or say ‘I don’t know.’” Maybe add a check—if the retrieved chunks don’t match well, it could just ask me to rephrase instead of making stuff up.

## Response for question
[View JSON File](response.json)


## Demo
![](https://github.com/ishika28/NLP_Assignment/blob/main/A6/demo_video.gif)