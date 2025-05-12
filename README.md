# YouTube-to-Blog Generator
Transform any YouTube video into a structured, markdown-formatted blog post-automatically.
<br> This project leverages advanced NLP pipelines (LangChain, Hugging Face, FAISS, and the Longformer Encoder-Decoder) to fetch transcripts, summarize content, and generate multi-section blog articles with titles and summaries.

## Features<br>
* Automatic Transcript Fetching: Just provide a YouTube URL.
* Intelligent Chunking: Handles long transcripts by splitting them into manageable pieces.
* Retrieval-Augmented Generation (RAG): Uses vector search (FAISS) to retrieve relevant context for each section.
* Long-Context Generation: Employs LED (Longformer Encoder-Decoder) for handling up to 16,384 tokens.
* Prompt Engineering: Custom prompts for title, outline, section content, and summary.
* Markdown Output: Generates a ready-to-publish blog post.

## Tech Stack
* Python

* LangChain (document loading, chunking, orchestration)

* Hugging Face Transformers (allenai/led-base-16384)

* sentence-transformers (for embeddings)

* FAISS (vector similarity search)

* dotenv (environment variable management)

## How It Works
#### Input:
* User provides a YouTube video URL.

#### Transcript Extraction:
* Uses LangChain’s YoutubeLoader to fetch the transcript.

#### Chunking:
* Transcript is split into context-aware chunks using RecursiveCharacterTextSplitter.

#### Embedding & Retrieval:
* Chunks are embedded with sentence-transformers and indexed in FAISS for context retrieval.

#### Blog Generation:

* Title: Generated via a concise, explicit prompt.

* Outline: SEO-friendly headings created from context.

* Sections: Each heading is expanded into a detailed section.

* Summary: Final summary generated from the full content.

#### Output:
* Markdown-formatted blog post, ready for editing or publishing.

### Sample Output
~~~
# India's Nuclear Journey: From Homi Bhabha to Today

### The Early Years of Nuclear Research
India’s nuclear program began in the 1940s under the visionary leadership of Dr. Homi Bhabha...

### Key Milestones and Global Impact
Major milestones include the peaceful nuclear explosion in 1974 and the Pokhran-II tests in 1998...

### Challenges and International Relations
India’s nuclear ambitions faced international scrutiny, sanctions, and complex geopolitics...

### The Role of Thorium and Indigenous Innovation
India’s focus on thorium-based reactors sets it apart in the global nuclear landscape...

### Looking Ahead: The Future of India’s Nuclear Program
With ongoing research and international cooperation, India aims for energy security and technological leadership...

**Summary:**  
India’s nuclear journey reflects decades of scientific ambition, resilience, and innovation. From Dr. Bhabha’s vision to modern advancements, India continues to shape its energy future.
~~~

## Challenges Faced

#### Token Limitations:
* Initially used Flan-T5-base, but hit 512-token input limits when processing long transcripts (some >90k tokens).

#### Chunking & Model Switching:
* Adopted strategic chunking and switched to the Longformer Encoder-Decoder (LED) for better long-context handling.

#### Repetitive & Incoherent Output:
* Early outputs were often repetitive or off-topic (sometimes just a single line repeated 100 times!).
* Improved coherence through iterative prompt engineering and chunk-wise generation.

#### No System Prompt Support:
* Unlike GPT-4, LED doesn’t support system prompts, so all instructions had to be embedded directly in the input prompts.

#### Manual Editing Still Needed:
* Even with improvements, some human tweaks are required for perfect output.

## Future Plans
* Smarter Chunk Handling:
* Summarize each chunk individually, then concatenate summaries for final generation to further reduce repetition and drift.

#### Model Upgrades:
* Experiment with GPT-4 or Perplexity API for higher-quality, more creative outputs.

#### Deployment:
* Build a simple web interface for public use and deploy the tool.

#### Continuous Iteration:
* Keep refining prompts, chunking strategies, and post-processing for better results.


## How to Use
* Clone the Repository & Install Dependencies
```
git clone https://github.com/mridulchdry17/youtube-to-blog-generator.git
cd youtube-to-blog-generator
pip install -r requirements.txt
```
### Set Up Environment Variables

* Create a .env file in the project root (if not already present).

* Add any required API keys or settings (if using external APIs or deployment).

### Run the Script

* Open your terminal and run:

```
python blog_generator.py
```

* When prompted (or in the code), paste the YouTube video URL you want to convert into a blog post.

#### Wait for Processing

* The tool will automatically:

* Fetch the transcript from YouTube.

* Chunk and embed the transcript.

* Retrieve relevant context and generate the blog title, headings, sections, and summary using the AI model.

* Assemble the full blog post in markdown format.

* View and Edit the Output

* The generated blog post will be displayed in your terminal.

* You can copy the output and paste it into your favorite blog editor (WordPress, Medium, etc.) or save it to a file.

#### Tip: Always review and proofread the AI-generated content before publishing, as some manual tweaks may be needed for clarity and accuracy.

### Example Usage
```
python blog_generator.py
```
### Input:

```
Enter YouTube URL: https://www.youtube.com/watch?v=gWwNZGRCDM4
```
### Output:

```
# India's Nuclear Journey: From Homi Bhabha to Today

### The Early Years of Nuclear Research
India’s nuclear program began in the 1940s under Dr. Homi Bhabha...

### Key Milestones and Global Impact
Major milestones include the peaceful nuclear explosion in 1974 and the Pokhran-II tests in 1998...

**Summary:**
India’s nuclear journey reflects decades of scientific ambition, resilience, and innovation.
```
