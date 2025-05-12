import os
from dotenv import load_dotenv
from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, pipeline

def download_transcript(youtube_url):
    """Download YouTube transcript using LangChain's loader"""
    loader = YoutubeLoader.from_youtube_url(youtube_url)
    docs = loader.load()
    return "\n".join(doc.page_content for doc in docs)

def setup_rag_pipeline(transcript_text):
    """Set up RAG with LED's tokenizer for long documents"""
    tokenizer = AutoTokenizer.from_pretrained("allenai/led-base-16384")
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer=tokenizer,
        chunk_size=4000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""]
    )
    chunks = text_splitter.split_text(transcript_text)
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 5})

def generate_blog_post(transcript_text):
    """Generate a high-quality, non-repetitive blog post using LED model"""
    load_dotenv()
    retriever = setup_rag_pipeline(transcript_text)

    model_id = "allenai/led-base-16384"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    pipe = pipeline(
        "text2text-generation",
        model=model_id,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.5,
        no_repeat_ngram_size=4,
        length_penalty=2.0
    )
    llm = HuggingFacePipeline(pipeline=pipe)

    # Improved prompt templates
    prompt_templates = {
        "title": PromptTemplate(
            template=(
                "You are an expert blog writer. Read the following context and generate an engaging, unique blog title. "
                "Also write the title in one or two lines only also Do NOT repeat words or phrases. Context:\n{context}\n\nTitle:"
            ),
            input_variables=["context", "question"]
        ),
        "outline": PromptTemplate(
            template=(
                "Based on the following context, write a concise, non-repetitive, SEO-friendly outline for a blog post. "
                "List 3-5 unique headings. Avoid repetition. Context:\n{context}\n\nOutline:"
            ),
            input_variables=["context", "question"]
        ),
        "section": PromptTemplate(
            template=(
                "Using the context and the outline heading, write a detailed, factually accurate, and non-repetitive blog section in 300 words "
                "Do NOT repeat sentences or phrases. Be clear, concise, and informative.\n\nContext: {context}\n\nHeading: {question}\n\nSection:"
            ),
            input_variables=["context", "question"]
        ),
        "summary": PromptTemplate(
            template=(
                "Summarize the following blog content in a clear, concise, and non-repetitive way and in 200 words only. Do NOT repeat sentences. Content:\n{context}\n\nSummary:"
            ),
            input_variables=["context", "question"]
        ),
    }

    # Helper to run a RAG chain
    def rag_chain(prompt_template, question):
        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt_template
            | llm
        )
        return chain.invoke(question).strip()

    # Generate title
    title = rag_chain(prompt_templates["title"], "Generate blog title")

    # Generate outline (headings)
    outline_text = rag_chain(prompt_templates["outline"], "Generate blog outline")
    headings = [h.strip("-• \n") for h in outline_text.split("\n") if h.strip()]

    # Generate each section using its heading
    sections = []
    for heading in headings:
        section = rag_chain(prompt_templates["section"], heading)
        sections.append(f"### {heading}\n{section}")

    # Generate summary from all sections
    full_content = "\n\n".join(sections)
    summary = rag_chain(prompt_templates["summary"], full_content)

   
    return (
    "# " + title + "\n\n"
    + "\n\n".join(sections) + "\n\n"
    + "**Summary:**\n" + summary
    )



if __name__ == "__main__":
    youtube_url = 'https://www.youtube.com/watch?v=gWwNZGRCDM4'
    transcript = download_transcript(youtube_url)
    blog = generate_blog_post(transcript)
    print("\nGenerated Blog:\n")
    print(blog)


# # 2nd best till now best for resume 
# import os
# from dotenv import load_dotenv
# from langchain.document_loaders import YoutubeLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import FAISS
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.prompts import PromptTemplate
# from langchain_core.runnables import RunnablePassthrough
# from langchain_huggingface import HuggingFacePipeline
# from transformers import AutoTokenizer, pipeline

# def download_transcript(youtube_url):
#     """Download YouTube transcript using LangChain's loader"""
#     loader = YoutubeLoader.from_youtube_url(youtube_url)
#     docs = loader.load()
#     return "\n".join(doc.page_content for doc in docs)

# def setup_rag_pipeline(transcript_text):
#     """Set up RAG with LED's tokenizer for long documents"""
#     # Use LED's tokenizer for proper chunking
#     tokenizer = AutoTokenizer.from_pretrained("allenai/led-base-16384")
    
#     text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
#         tokenizer=tokenizer,
#         chunk_size=4000,  # Increased for longer sequences
#         chunk_overlap=200,
#         separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""]
#     )
    
#     chunks = text_splitter.split_text(transcript_text)
    
#     embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
#     vectorstore = FAISS.from_texts(chunks, embeddings)
#     return vectorstore.as_retriever(search_kwargs={"k": 5})

# def generate_blog_post(transcript_text):
#     """Generate blog post using LED model"""
#     load_dotenv()
#     retriever = setup_rag_pipeline(transcript_text)

#     # LED-specific configuration
#     model_id = "allenai/led-base-16384"
#     tokenizer = AutoTokenizer.from_pretrained(model_id)
    
#     pipe = pipeline(
#         "text2text-generation",
#         model=model_id,
#         tokenizer=tokenizer,
#         max_new_tokens=512,  # Increased for longer outputs
#         temperature=0.7,
#         no_repeat_ngram_size=3,
#         length_penalty=2.0
#     )

#     llm = HuggingFacePipeline(pipeline=pipe)

#     # RAG chain setup
#     def create_rag_chain(template):
#         return (
#             {"context": retriever, "question": RunnablePassthrough()}
#             | PromptTemplate(template=template, input_variables=["context", "question"])
#             | llm
#         )

#     # Define chains
#     chains = {
#         "title": create_rag_chain("Generate blog title: {context}\nTitle:"),
#         "headings": create_rag_chain("Create 3-5 headings: {context}\nHeadings:"),
#         "content": create_rag_chain("Write detailed content: {context}\nContent:"),
#         "summary": create_rag_chain("Summarize key points: {context}\nSummary:")
#     }

#     # Execute chains 
#     results = {name: chain.invoke(name) for name, chain in chains.items()}

#     return (
#         f"# {results['title'].strip()}\n\n"
#         f"## {results['headings'].strip()}\n\n"
#         f"{results['content'].strip()}\n\n"
#         f"**Summary:**\n{results['summary'].strip()}"
#     )
#     # # Execute chains
#     # results = {name: chain.invoke(task) for name, (task, chain) in zip(
#     #     ["Title Generation", "Headings Creation", "Content Writing", "Summary Creation"],
#     #     chains.items()
#     # )}

#     # return f"# {results['title'].strip()}\n\n## {results['headings'].strip()}\n\n{results['content'].strip()}\n\n**Summary:**\n{results['summary'].strip()}"

# if __name__ == "__main__":
#     youtube_url = 'https://www.youtube.com/watch?v=gWwNZGRCDM4'
#     transcript = download_transcript(youtube_url)
#     blog = generate_blog_post(transcript)
#     print("\nGenerated Blog:\n")
#     print(blog)


## 3rd best model but having issue on text generation becox of context limit 
# import os
# import glob
# import sys
# import subprocess
# import whisper
# import yt_dlp
# from langchain_huggingface import HuggingFacePipeline
# from transformers import AutoTokenizer, pipeline
# from langchain.prompts import PromptTemplate
# from langchain_core.runnables import RunnablePassthrough
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import FAISS
# from langchain.embeddings import HuggingFaceEmbeddings

# from dotenv import load_dotenv


# def download_transcript(youtube_url):
#     command = [
#         sys.executable, "-m", "yt_dlp", 
#         "--write-auto-sub",
#         "--sub-format", "vtt",
#         "--skip-download",
#         youtube_url,
#         "-o", "%(title)s.%(ext)s"
#     ]
    
#     subprocess.run(command, check=True)

#     vtt_files = glob.glob("*.vtt")
#     if not vtt_files:
#         raise FileNotFoundError("No .vtt subtitle file found.")

#     with open(vtt_files[0], "r", encoding="utf-8") as f:
#         transcript_text = f.read()

#     return transcript_text

# def setup_rag_pipeline(transcript_text):
#     """
#     Set up the RAG pipeline with FAISS and HuggingFace embeddings.
#     """
#     tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")

#     text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
#         tokenizer=tokenizer,
#         chunk_size=400,
#         chunk_overlap=40,
#     )
    
#     text_splitter.separators = ["\n\n", "\n", ".", "?", "!", " ", ""]

#     chunks = text_splitter.split_text(transcript_text)

#     embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')
#     vectorstore = FAISS.from_texts(chunks,embeddings)
#     return vectorstore.as_retriever(search_kwargs={"k": 5})



# def transcribe_audio(video_file):
#     print("Transcribing audio...")
#     model = whisper.load_model("base")
#     result = model.transcribe(video_file)
#     return result['text']

# def generate_blog_post(transcript_text):
#     """
#     Generate a blog post from the transcript text using LangChain.
#     """

#     print("Generating blog post...")
#     load_dotenv()

#     retriever = setup_rag_pipeline(transcript_text)

#     # Modern pipeline initialization
#     model_id = "google/flan-t5-base"
#     tokenizer = AutoTokenizer.from_pretrained(model_id)
    
#     pipe = pipeline(
#         "text2text-generation",
#         model=model_id,
#         tokenizer=tokenizer,
#         max_new_tokens=300,
#         temperature=0.6
#     )
    
#     llm = HuggingFacePipeline(pipeline=pipe)

#     # Create base chain components
#     prompt_template = lambda template: PromptTemplate(
#         template=template,
#         input_variables = ['context','question']
#     )

#     # Generic Rag chain Constructor 
#     def create_rag_chain(template):
#         return (
#             {'context':retriever,'question':RunnablePassthrough()}
#             | prompt_template(template)
#             | llm
#         )
    
#     # create individual chains
#     title_chain = create_rag_chain(
#         "Generate an engaging blog title using this context:\n{context}\n\nTitle:"
#     )

#     headings_chain = create_rag_chain(
#         "Generate 3-5 SEO-friendly blog headings using this context:\n{context}\n\nHeadings:"
#     )
 
#     content_chain = create_rag_chain(
#         "Generate detailed blog content using this context:\n{context}\n\nContent:"
#     )
#     summary_chain = create_rag_chain(
#         "Summarize this context for a blog post:\n{context}\n\nSummary:"
#     )

#     # Execute chains
#     title = title_chain.invoke("Generate blog title")
#     headings = headings_chain.invoke("Generate blog headings")
#     content = content_chain.invoke("Generate blog content")
#     summary = summary_chain.invoke("Generate summary")
    
#     return f"# {title.strip()}\n\n{headings.strip()}\n\n{content.strip()}\n\n**Summary:**\n{summary.strip()}"


# if __name__ == "__main__":
#     youtube_url = 'https://www.youtube.com/watch?v=gWwNZGRCDM4'
#     transcript_text = download_transcript(youtube_url)
#     blog_post = generate_blog_post(transcript_text)
#     print("\nBlog Generated:\n")
#     print(blog_post)

# import os
# import glob
# import subprocess
# import torch
# from langchain_huggingface import HuggingFacePipeline
# from transformers import AutoTokenizer, pipeline, BitsAndBytesConfig
# from langchain.prompts import PromptTemplate
# from langchain_core.runnables import RunnablePassthrough
# from dotenv import load_dotenv
# import yt_dlp
# import sys

# def download_transcript(youtube_url):
#     """Download YouTube subtitles and return as text"""
#     command = [
#         sys.executable, "-m", "yt_dlp",
#         "--write-auto-sub",
#         "--sub-format", "vtt",
#         "--skip-download",
#         youtube_url,
#         "-o", "%(title)s.%(ext)s"
#     ]
    
#     try:
#         subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
#     except subprocess.CalledProcessError as e:
#         raise RuntimeError(f"Failed to download subtitles: {str(e)}")

#     vtt_files = glob.glob("*.vtt")
#     if not vtt_files:
#         raise FileNotFoundError("No subtitle file found")

#     with open(vtt_files[0], "r", encoding="utf-8") as f:
#         content = f.read()
    
#     os.remove(vtt_files[0])  # Cleanup subtitle file
#     return content

# def chunk_text(text, model_name="google/flan-t5-base", max_tokens=400):
#     """Split text into context-preserving chunks"""
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     sentences = text.split('. ')
#     chunks = []
#     current_chunk = []
#     current_length = 0
    
#     for sentence in sentences:
#         sentence_tokens = tokenizer.encode(sentence, add_special_tokens=False)
#         if current_length + len(sentence_tokens) > max_tokens:
#             chunks.append('. '.join(current_chunk) + '.')
#             current_chunk = [sentence]
#             current_length = len(sentence_tokens)
#         else:
#             current_chunk.append(sentence)
#             current_length += len(sentence_tokens)
    
#     if current_chunk:
#         chunks.append('. '.join(current_chunk) + '.')
    
#     return chunks

# def initialize_model():
#     """Initialize model with proper configuration"""
#     quantization_config = BitsAndBytesConfig(
#         load_in_8bit=True if torch.cuda.is_available() else False,
#         llm_int8_skip_modules=["final_layer_norm"]
#     )

#     try:
#         return pipeline(
#             "text2text-generation",
#             model="google/flan-t5-base",
#             tokenizer=AutoTokenizer.from_pretrained("google/flan-t5-base"),
#             device_map="auto" if torch.cuda.is_available() else None,
#             quantization_config=quantization_config,
#             max_new_tokens=256,
#             temperature=0.6,
#             model_kwargs={"cache_dir": "model_cache"}
#         )
#     except Exception as e:
#         print(f"Using CPU-only mode: {str(e)}")
#         return pipeline(
#             "text2text-generation",
#             model="google/flan-t5-base",
#             max_new_tokens=128,
#             temperature=0.4
#         )

# def generate_blog_content(transcript_text):
#     """Generate blog content from transcript"""
#     load_dotenv()
    
#     # Initialize components
#     chunks = chunk_text(transcript_text)
#     pipe = initialize_model()
#     llm = HuggingFacePipeline(pipeline=pipe)

#     # Process chunks
#     summarized_chunks = []
#     for chunk in chunks:
#         try:
#             result = llm.invoke(f"Summarize this content: {chunk[:3000]}")
#             summarized_chunks.append(result.strip())
#         except Exception as e:
#             print(f"Error processing chunk: {str(e)}")
    
#     final_content = "\n".join(summarized_chunks)

#     # Define blog components
#     title_chain = PromptTemplate.from_template(
#         "Generate a blog title about: {content}"
#     ) | llm
    
#     headings_chain = PromptTemplate.from_template(
#         "Create 3-5 SEO headings for content: {content}"
#     ) | llm
    
#     content_chain = PromptTemplate.from_template(
#         "Write detailed blog post about: {content}"
#     ) | llm

#     # Generate final content
#     try:
#         return {
#             "title": title_chain.invoke({"content": final_content}),
#             "headings": headings_chain.invoke({"content": final_content}),
#             "content": content_chain.invoke({"content": final_content})
#         }
#     except Exception as e:
#         raise RuntimeError(f"Generation failed: {str(e)}")

# def format_blog(post_data):
#     """Format generated content into markdown"""
#     return f"""# {post_data['title']}

# ## Key Sections
# {post_data['headings']}

# {post_data['content']}
# """

# if __name__ == "__main__":
#     try:
#         transcript = download_transcript('https://www.youtube.com/watch?v=gWwNZGRCDM4')
#         blog_data = generate_blog_content(transcript)
#         print("\nGenerated Blog:\n")
#         print(format_blog(blog_data))
#     except Exception as e:
#         print(f"Error: {str(e)}")
#         sys.exit(1)
