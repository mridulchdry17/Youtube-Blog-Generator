# 🎥📄 YouTube to Blog Generator

A Flask-based web app that **generates structured blog posts** from any YouTube video URL using **LangChain**, **Groq (Gemma2-9b-it)**, and **LLM prompt engineering**. Ideal for turning educational or technical videos into well-formed written content.

---

## 🌐  Demo
Watch this video to see **how this project works**:  
👉 [Project Demo Video](https://www.youtube.com/watch?v=8P23ce9TGGw)


---

## 🛠️ Tech Stack

- **Flask** – for backend web routing
- **LangChain** – to load and process YouTube transcripts
- **Groq (Gemma2-9b-it)** – as the LLM to generate content
- **RecursiveCharacterTextSplitter** – for chunking long transcripts
- **HTML/CSS** – for the landing and input pages

---

## ⚙️ How It Works

1. **User submits YouTube URL** via frontend
2. **Transcript is fetched** using `YoutubeLoader`
3. Transcript is **split into overlapping chunks**
4. Each chunk is summarized using the LLM
5. Summaries are combined into a **condensed context**
6. LLM is prompted to:
   - Generate a **title**
   - Generate an **outline with subheadings**
   - Iterate through each subheading and create content
   - Generate a **summary**
7. Final blog is returned in Markdown format with a **CTA**

---

## 🌟 Highlights & Thought Process

- ✅ **Efficient Processing**: Instead of feeding the full transcript directly, it's split into manageable chunks for clarity and performance.
- ✅ **Stored Intermediate Summaries**: Each chunk summary is stored and reused in the full blog context, optimizing API usage and maintaining coherence.
- ✅ **Iterative Section Writing**: A `for` loop generates content for **each heading** in the outline. This ensures structure, relevance, and flow throughout the blog.
- ✅ **Clean Prompt Engineering**: Carefully crafted prompts extract only the required content (title, sections, summary) with no extra fluff.
- ✅ **Markdown-Based Output**: The result is well-structured Markdown, ready for publishing.
- ✅ **Deployed on Render**: Easy to access, fast, and reliable hosting.

---

## 📁 File Structure

```bash
.
├── app.py                  # Flask server
├── templates/
│   ├── landing.html        # Home/landing page
│   └── index.html          # Blog generator form
├── requirements.txt
└── README.md
```

### RUN LOCALLY 
```
git clone https://github.com/mridulchdry17/youtube-blog-generator.git
cd youtube-blog-generator
pip install -r requirements.txt

# Add your GROQ_API_KEY in a .env file
touch .env
echo "GROQ_API_KEY=your_api_key_here" > .env

# Run the server
python app.py
```

## 🙌 Future Ideas
1) Add support for multi-language transcripts

2) Style the blog using HTML rendering

3) Export blog as PDF or publish to Medium
