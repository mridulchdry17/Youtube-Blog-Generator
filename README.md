# YouTube to Blog Generator

A web application that converts YouTube videos into well-structured blog posts using AI. The application uses Flask for the backend, modern UI with Tailwind CSS, and Ollama for AI text generation.

## Features

- Convert YouTube videos to blog posts
- Modern, responsive UI
- Real-time blog post generation
- Copy to clipboard functionality
- Loading states and error handling

## Prerequisites

- Python 3.8 or higher
- Ollama installed and running locally
- pip (Python package manager)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd youtube-to-blog
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Make sure Ollama is running and the gemma:2b model is available:
```bash
ollama pull gemma:2b
```

## Running the Application

1. Start the Flask application:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

## Usage

1. Enter a YouTube video URL in the input field
2. Click "Generate Blog Post"
3. Wait for the blog post to be generated
4. The generated blog post will appear below the form
5. Use the "Copy to Clipboard" button to copy the content

## Note

Make sure you have Ollama running locally with the gemma:2b model installed. The application requires an active internet connection to fetch YouTube video transcripts.