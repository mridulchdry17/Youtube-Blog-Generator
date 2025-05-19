from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from groq import Groq
import re
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def load_model(model_name="gemma2-9b-it"):
    print(f"Using model: {model_name}")
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return client

def generate_text(model_name: str, prompt: str) -> str:
    """Generate text using the Groq model with streaming."""
    print("Generating response from model...")
    response = model_name.chat.completions.create(
        model="gemma2-9b-it",
        messages=[
            {"role": "user", "content": prompt}
        ],
        stream=True
    )
    
    full_response = ""
    for chunk in response:
        if chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            print(content, end="", flush=True)
            full_response += content
    
    print("\n")  # Add a newline after streaming
    return full_response


def download_transcript(youtube_url: str) -> str:
    """Download transcript from YouTube video URL."""
    loader = YoutubeLoader.from_youtube_url(youtube_url)
    docs = loader.load()
    transcript = "\n".join(doc.page_content for doc in docs)
    return transcript


def split_text_into_chunks(text: str, chunk_size=2000, chunk_overlap=200) -> list:
    """Split large text into smaller overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_text(text)
    return chunks


def clean_section(text):
    # Remove generic phrases
    text = re.sub(r"^Sure,? (here( is| are))?[^\n]*:?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^Here( is| are)?[^\n]*:?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^Here's a 300-word blog section under the given heading:?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^Here is a 300-word blog section under the given heading:?\s*", "", text, flags=re.IGNORECASE)
    return text.strip()


def generate_blog_post(model_name: str, transcript_text: str) -> str:
    """Generate a concise blog post from transcript text."""
    prompt_templates = {
        "title": (
            "You are a professional blog writer. Generate ONLY a title, no explanations or options. "
            "Create a clear, engaging blog title that captures the main value. "
            "Focus on the key benefit or solution. "
            "Keep it under 60 characters.\n\n{context}\n\nTitle:"
        ),
        "outline": (
            "You are a professional blog writer. Generate ONLY the headings, no explanations or options. "
            "Create 5 clear headings that tell a story:\n"
            "1. Introduction (hook and context)\n"
            "2. The Challenge (what's the problem)\n"
            "3. The Solution (how to solve it)\n"
            "4. Key Takeaways (main points to remember)\n"
            "5. Conclusion (what's next)\n\n"
            "Keep headings clear and direct.\n\n{context}\n\nOutline:"
        ),
        "section": (
            "You are a professional blog writer. Generate ONLY the section content, no explanations or options. "
            "Write a concise section for '{question}'. "
            "Keep it clear and practical. "
            "Use simple examples where helpful. "
            "Aim for 2-3 paragraphs.\n\n"
            "Context:\n{context}\n\nSection:"
        ),
        "summary": (
            "You are a professional blog writer. Generate ONLY the summary, no explanations or options. "
            "Write a brief 100-word summary that captures the main points. "
            "Keep it simple and actionable.\n\n"
            "Content:\n\n{context}\n\nSummary:"
        ),
        "chunk_summary": (
            "You are a professional blog writer. Generate ONLY the summary, no explanations or options. "
            "Extract the key points and main ideas. "
            "Focus on practical insights.\n\n{context}\n\nSummary:"
        )
    }

    chunks = split_text_into_chunks(transcript_text, chunk_size=1500, chunk_overlap=150)
    print("Generating blog post...")

    summaries = []
    for chunk in chunks:
        prompt = prompt_templates["chunk_summary"].format(context=chunk)
        summary = generate_text(model_name, prompt)
        summaries.append(summary.strip())

    condensed_context = "\n\n".join(summaries)

    def generate_for(template, context, question=""):
        prompt = template.format(context=context, question=question)
        return generate_text(model_name, prompt).strip()

    title = generate_for(prompt_templates["title"], condensed_context)
    outline = generate_for(prompt_templates["outline"], condensed_context)
    headings = [line.strip("-• \n") for line in outline.split("\n") if line.strip()]

    sections = []
    for heading in headings:
        section = generate_for(prompt_templates["section"], condensed_context, heading)
        section = clean_section(section)
        sections.append(f"### {heading}\n{section}")

    full_content = "\n\n".join(sections)
    summary = generate_for(prompt_templates["summary"], full_content)

    # Simple, clear CTA with option for another blog
    cta_section = (
        "\n\n---\n\n"
        "## Ready to Learn More?\n\n"
        "1. Share this article with your network\n"
        "2. Leave a comment with your thoughts\n"
        "3. Subscribe for more insights\n\n"
        "Would you like to generate a blog post on another topic? Just provide a YouTube video URL!"
    )

    blog_post = (
        f"# {title}\n\n"
        + full_content
        + f"\n\n**Summary:**\n{summary}"
        + cta_section
    )

    return blog_post 