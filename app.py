from flask import Flask, render_template, request, jsonify
from blog_gen import download_transcript, generate_blog_post, load_model
import os

app = Flask(__name__)
model = load_model()

@app.route('/')
def home():
    return render_template('landing.html')

@app.route('/generate')
def generate_page():
    return render_template('index.html')

@app.route('/api/generate', methods=['POST'])
def generate():
    try:
        youtube_url = request.json.get('youtube_url')
        if not youtube_url:
            return jsonify({'error': 'YouTube URL is required'}), 400

        # Download transcript
        transcript = download_transcript(youtube_url)
        
        # Generate blog post
        blog_post = generate_blog_post(model, transcript)
        
        return jsonify({
            'success': True,
            'blog_post': blog_post
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)), debug=True)
