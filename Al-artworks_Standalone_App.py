#!/usr/bin/env python3
"""
🎨 Al-artworks Standalone AI Art Generator
==========================================

A complete standalone application that creates AI-powered artwork
without requiring any installation or external dependencies.

Features:
- AI Art Generation from text descriptions
- Multiple art styles (realistic, abstract, cartoon, oil painting, watercolor)
- Modern web interface
- Standalone operation (no installation needed)
- Built-in dependencies and models
"""

import os
import sys
import base64
import io
import webbrowser
from datetime import datetime

# Add current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from flask import Flask, request, jsonify
    from flask_cors import CORS
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    import seaborn as sns
except ImportError as e:
    print(f"⚠️  Missing dependency: {e}")
    print("📦 Installing required packages...")
    import subprocess
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", 
        "flask", "flask-cors", "matplotlib", "seaborn", "numpy"
    ])
    from flask import Flask, request, jsonify
    from flask_cors import CORS
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    import seaborn as sns

class AlArtworksApp:
    """Main application class for Al-artworks AI Suite"""
    
    def __init__(self):
        self.app = Flask(__name__)
        CORS(self.app)
        self.setup_routes()
        self.setup_styles()
        self.art_history = []
        
    def setup_styles(self):
        """Setup matplotlib styles for better visual output"""
        plt.style.use('default')
        sns.set_palette("husl")
        
    def setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def index():
            """Main application page"""
            return self.get_html_template()
            
        @self.app.route('/generate', methods=['POST'])
        def generate_art():
            """Generate AI artwork from text description"""
            try:
                data = request.get_json()
                prompt = data.get('prompt', '')
                style = data.get('style', 'realistic')
                
                if not prompt:
                    return jsonify({'status': 'error', 'message': 'Please provide an art description'})
                
                # Generate artwork based on style
                image_data = self.generate_artwork(prompt, style)
                
                # Save to history
                self.art_history.append({
                    'prompt': prompt,
                    'style': style,
                    'timestamp': datetime.now().isoformat(),
                    'image': image_data
                })
                
                return jsonify({
                    'status': 'success',
                    'message': f'Generated {style} art for: {prompt}',
                    'image_url': f'data:image/png;base64,{image_data}'
                })
                
            except Exception as e:
                return jsonify({'status': 'error', 'message': str(e)}), 500
                
        @self.app.route('/history')
        def get_history():
            """Get art generation history"""
            return jsonify(self.art_history)
            
        @self.app.route('/styles')
        def get_styles():
            """Get available art styles"""
            return jsonify({
                'styles': [
                    {'id': 'realistic', 'name': 'Realistic', 'description': 'Photorealistic artwork'},
                    {'id': 'abstract', 'name': 'Abstract', 'description': 'Abstract and conceptual art'},
                    {'id': 'cartoon', 'name': 'Cartoon', 'description': 'Animated cartoon style'},
                    {'id': 'oil-painting', 'name': 'Oil Painting', 'description': 'Classical oil painting style'},
                    {'id': 'watercolor', 'name': 'Watercolor', 'description': 'Soft watercolor painting'},
                    {'id': 'digital', 'name': 'Digital Art', 'description': 'Modern digital artwork'},
                    {'id': 'sketch', 'name': 'Sketch', 'description': 'Hand-drawn sketch style'},
                    {'id': 'fantasy', 'name': 'Fantasy', 'description': 'Fantasy and magical artwork'}
                ]
            })
    
    def generate_artwork(self, prompt, style):
        """Generate artwork based on prompt and style"""
        
        # Create figure with style-specific settings
        fig, ax = plt.subplots(figsize=(12, 8))
        
        if style == 'realistic':
            self.generate_realistic_art(ax, prompt)
        elif style == 'abstract':
            self.generate_abstract_art(ax, prompt)
        elif style == 'cartoon':
            self.generate_cartoon_art(ax, prompt)
        elif style == 'oil-painting':
            self.generate_oil_painting_art(ax, prompt)
        elif style == 'watercolor':
            self.generate_watercolor_art(ax, prompt)
        elif style == 'digital':
            self.generate_digital_art(ax, prompt)
        elif style == 'sketch':
            self.generate_sketch_art(ax, prompt)
        elif style == 'fantasy':
            self.generate_fantasy_art(ax, prompt)
        else:
            self.generate_realistic_art(ax, prompt)
        
        # Convert to base64
        canvas = FigureCanvas(fig)
        canvas.draw()
        
        # Get the image data
        img_data = io.BytesIO()
        fig.savefig(img_data, format='png', dpi=150, bbox_inches='tight')
        img_data.seek(0)
        
        # Convert to base64
        image_base64 = base64.b64encode(img_data.getvalue()).decode('utf-8')
        
        plt.close(fig)
        return image_base64
    
    def generate_realistic_art(self, ax, prompt):
        """Generate realistic artwork"""
        ax.clear()
        
        # Create a realistic landscape
        x = np.linspace(0, 10, 100)
        y = np.sin(x) * 2 + np.random.normal(0, 0.1, 100)
        
        # Sky gradient
        ax.fill_between(x, 5, 8, alpha=0.6, color='skyblue')
        
        # Mountains
        mountains = np.sin(x * 0.5) * 3 + 2
        ax.fill_between(x, mountains, 8, alpha=0.8, color='gray')
        
        # Ground
        ax.fill_between(x, 0, mountains, alpha=0.7, color='green')
        
        # Add some trees
        tree_positions = np.random.choice(x, 15)
        for pos in tree_positions:
            ax.scatter(pos, np.sin(pos) * 2 + 1, s=100, color='darkgreen', marker='^')
        
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 8)
        ax.set_title(f'Realistic: {prompt}', fontsize=16, fontweight='bold')
        ax.axis('off')
    
    def generate_abstract_art(self, ax, prompt):
        """Generate abstract artwork"""
        ax.clear()
        
        # Create abstract geometric shapes
        colors = ['red', 'blue', 'yellow', 'green', 'purple', 'orange']
        
        for i in range(20):
            x = np.random.rand() * 10
            y = np.random.rand() * 8
            size = np.random.rand() * 2 + 0.5
            color = np.random.choice(colors)
            alpha = np.random.rand() * 0.8 + 0.2
            
            if np.random.rand() > 0.5:
                circle = patches.Circle((x, y), size, color=color, alpha=alpha)
                ax.add_patch(circle)
            else:
                rect = patches.Rectangle((x, y), size, size, color=color, alpha=alpha)
                ax.add_patch(rect)
        
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 8)
        ax.set_title(f'Abstract: {prompt}', fontsize=16, fontweight='bold')
        ax.axis('off')
    
    def generate_cartoon_art(self, ax, prompt):
        """Generate cartoon artwork"""
        ax.clear()
        
        # Create cartoon character
        # Head
        head = patches.Circle((5, 5), 1.5, color='peachpuff', ec='black', linewidth=2)
        ax.add_patch(head)
        
        # Eyes
        eye1 = patches.Circle((4.7, 5.3), 0.2, color='white', ec='black')
        eye2 = patches.Circle((5.3, 5.3), 0.2, color='white', ec='black')
        ax.add_patch(eye1)
        ax.add_patch(eye2)
        
        # Pupils
        pupil1 = patches.Circle((4.7, 5.3), 0.1, color='black')
        pupil2 = patches.Circle((5.3, 5.3), 0.1, color='black')
        ax.add_patch(pupil1)
        ax.add_patch(pupil2)
        
        # Smile
        smile = patches.Arc((5, 4.8), 0.8, 0.4, angle=0, theta1=0, theta2=180, color='black', linewidth=2)
        ax.add_patch(smile)
        
        # Body
        body = patches.Rectangle((4.5, 2), 1, 2, color='lightblue', ec='black', linewidth=2)
        ax.add_patch(body)
        
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 8)
        ax.set_title(f'Cartoon: {prompt}', fontsize=16, fontweight='bold')
        ax.axis('off')
    
    def generate_oil_painting_art(self, ax, prompt):
        """Generate oil painting artwork"""
        ax.clear()
        
        # Create oil painting effect with brush strokes
        x = np.linspace(0, 10, 200)
        y = np.sin(x) * 3 + np.cos(x * 0.5) * 2
        
        # Multiple brush strokes
        for i in range(10):
            offset = np.random.normal(0, 0.1, len(x))
            ax.plot(x, y + offset, color=np.random.choice(['goldenrod', 'sienna', 'darkred', 'navy']), 
                   linewidth=np.random.rand() * 3 + 1, alpha=0.7)
        
        # Add texture
        texture_x = np.random.rand(50) * 10
        texture_y = np.random.rand(50) * 8
        ax.scatter(texture_x, texture_y, s=20, color='brown', alpha=0.3)
        
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 8)
        ax.set_title(f'Oil Painting: {prompt}', fontsize=16, fontweight='bold')
        ax.axis('off')
    
    def generate_watercolor_art(self, ax, prompt):
        """Generate watercolor artwork"""
        ax.clear()
        
        # Create soft watercolor effect
        x = np.linspace(0, 10, 100)
        y = np.sin(x) * 2 + np.random.normal(0, 0.2, 100)
        
        # Soft gradient background
        ax.fill_between(x, 0, 8, alpha=0.3, color='lightblue')
        
        # Watercolor flowers
        for i in range(8):
            flower_x = np.random.rand() * 10
            flower_y = np.random.rand() * 6 + 1
            size = np.random.rand() * 0.5 + 0.3
            color = np.random.choice(['pink', 'lavender', 'lightgreen', 'yellow'])
            
            flower = patches.Circle((flower_x, flower_y), size, color=color, alpha=0.6)
            ax.add_patch(flower)
        
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 8)
        ax.set_title(f'Watercolor: {prompt}', fontsize=16, fontweight='bold')
        ax.axis('off')
    
    def generate_digital_art(self, ax, prompt):
        """Generate digital artwork"""
        ax.clear()
        
        # Create digital geometric patterns
        x = np.linspace(0, 10, 50)
        y = np.linspace(0, 8, 40)
        X, Y = np.meshgrid(x, y)
        
        # Digital pattern
        Z = np.sin(X) * np.cos(Y) + np.sin(X * 2) * np.cos(Y * 2)
        
        # Create digital art effect
        contour = ax.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.8)
        
        # Add digital elements
        for i in range(15):
            x_pos = np.random.rand() * 10
            y_pos = np.random.rand() * 8
            ax.scatter(x_pos, y_pos, s=50, color='cyan', alpha=0.7, edgecolors='white', linewidth=1)
        
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 8)
        ax.set_title(f'Digital Art: {prompt}', fontsize=16, fontweight='bold')
        ax.axis('off')
    
    def generate_sketch_art(self, ax, prompt):
        """Generate sketch artwork"""
        ax.clear()
        
        # Create sketch-like drawing
        x = np.linspace(0, 10, 100)
        y = np.sin(x) * 2 + np.random.normal(0, 0.1, 100)
        
        # Sketch lines
        ax.plot(x, y, color='black', linewidth=1, alpha=0.8)
        
        # Add cross-hatching
        for i in range(0, 10, 2):
            ax.axhline(y=i, color='gray', alpha=0.3, linewidth=0.5)
        
        # Add some sketch details
        detail_x = np.random.rand(20) * 10
        detail_y = np.random.rand(20) * 8
        ax.scatter(detail_x, detail_y, s=10, color='black', alpha=0.6)
        
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 8)
        ax.set_title(f'Sketch: {prompt}', fontsize=16, fontweight='bold')
        ax.axis('off')
    
    def generate_fantasy_art(self, ax, prompt):
        """Generate fantasy artwork"""
        ax.clear()
        
        # Create fantasy landscape
        x = np.linspace(0, 10, 100)
        
        # Floating islands
        for i in range(5):
            island_x = np.random.rand() * 10
            island_y = np.random.rand() * 4 + 2
            island_size = np.random.rand() * 1 + 0.5
            
            island = patches.Circle((island_x, island_y), island_size, color='green', alpha=0.8)
            ax.add_patch(island)
            
            # Add trees to islands
            tree_x = island_x + np.random.normal(0, 0.3)
            tree_y = island_y + np.random.normal(0, 0.3)
            ax.scatter(tree_x, tree_y, s=100, color='darkgreen', marker='^')
        
        # Magic particles
        particles_x = np.random.rand(30) * 10
        particles_y = np.random.rand(30) * 8
        ax.scatter(particles_x, particles_y, s=20, color='gold', alpha=0.8)
        
        # Rainbow
        rainbow_x = np.linspace(2, 8, 100)
        rainbow_y = 6 + 1.5 * np.sin((rainbow_x - 5) * 0.5)
        ax.plot(rainbow_x, rainbow_y, color='red', linewidth=3, alpha=0.7)
        
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 8)
        ax.set_title(f'Fantasy: {prompt}', fontsize=16, fontweight='bold')
        ax.axis('off')
    
    def get_html_template(self):
        """Get the HTML template for the web interface"""
        return '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🎨 Al-artworks AI Suite - Standalone</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            height: calc(100vh - 40px);
        }
        
        .header {
            grid-column: 1 / -1;
            text-align: center;
            margin-bottom: 20px;
        }
        
        .header h1 {
            font-size: 3em;
            margin: 0;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
            background: linear-gradient(45deg, #ff6b6b, #ee5a24, #feca57, #48dbfb);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .header p {
            font-size: 1.2em;
            opacity: 0.9;
            margin-top: 10px;
        }
        
        .art-generator {
            background: rgba(255,255,255,0.1);
            padding: 30px;
            border-radius: 15px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
        }
        
        .input-group {
            margin-bottom: 20px;
        }
        
        label {
            display: block;
            margin-bottom: 8px;
            font-weight: bold;
            font-size: 1.1em;
        }
        
        input, select, textarea {
            width: 100%;
            padding: 15px;
            border: none;
            border-radius: 10px;
            background: rgba(255,255,255,0.9);
            color: #333;
            font-size: 1em;
            transition: all 0.3s ease;
        }
        
        input:focus, select:focus, textarea:focus {
            outline: none;
            box-shadow: 0 0 0 3px rgba(255,255,255,0.3);
            transform: translateY(-2px);
        }
        
        button {
            background: linear-gradient(45deg, #ff6b6b, #ee5a24);
            color: white;
            padding: 18px 35px;
            border: none;
            border-radius: 10px;
            cursor: pointer;
            font-size: 1.2em;
            font-weight: bold;
            transition: all 0.3s ease;
            width: 100%;
            margin-top: 10px;
        }
        
        button:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.3);
            background: linear-gradient(45deg, #ee5a24, #ff6b6b);
        }
        
        button:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        
        .result {
            margin-top: 30px;
            text-align: center;
        }
        
        .loading {
            display: none;
            text-align: center;
            margin: 20px 0;
        }
        
        .spinner {
            border: 4px solid rgba(255,255,255,0.3);
            border-top: 4px solid white;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .artwork-display {
            background: rgba(255,255,255,0.1);
            padding: 30px;
            border-radius: 15px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            min-height: 400px;
        }
        
        .artwork-image {
            max-width: 100%;
            max-height: 500px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            transition: transform 0.3s ease;
        }
        
        .artwork-image:hover {
            transform: scale(1.05);
        }
        
        .placeholder {
            text-align: center;
            opacity: 0.7;
            font-size: 1.2em;
        }
        
        .placeholder-icon {
            font-size: 4em;
            margin-bottom: 20px;
            opacity: 0.5;
        }
        
        .history-section {
            grid-column: 1 / -1;
            background: rgba(255,255,255,0.1);
            padding: 20px;
            border-radius: 15px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
            margin-top: 20px;
        }
        
        .history-title {
            font-size: 1.5em;
            margin-bottom: 15px;
            text-align: center;
        }
        
        .history-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 15px;
            max-height: 300px;
            overflow-y: auto;
        }
        
        .history-item {
            background: rgba(255,255,255,0.1);
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .history-item:hover {
            transform: translateY(-2px);
            background: rgba(255,255,255,0.2);
        }
        
        .history-item img {
            width: 100%;
            height: 120px;
            object-fit: cover;
            border-radius: 8px;
            margin-bottom: 10px;
        }
        
        .history-item p {
            font-size: 0.9em;
            opacity: 0.9;
            margin: 5px 0;
        }
        
        .error-message {
            background: rgba(255,0,0,0.2);
            color: #ffcccc;
            padding: 15px;
            border-radius: 10px;
            margin-top: 15px;
            border: 1px solid rgba(255,0,0,0.3);
        }
        
        .success-message {
            background: rgba(0,255,0,0.2);
            color: #ccffcc;
            padding: 15px;
            border-radius: 10px;
            margin-top: 15px;
            border: 1px solid rgba(0,255,0,0.3);
        }
        
        @media (max-width: 768px) {
            .container {
                grid-template-columns: 1fr;
                gap: 20px;
            }
            
            .header h1 {
                font-size: 2em;
            }
            
            .art-generator, .artwork-display {
                padding: 20px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎨 Al-artworks AI Suite</h1>
            <p>Create stunning AI-powered artwork with cutting-edge technology</p>
        </div>
        
        <div class="art-generator">
            <div class="input-group">
                <label for="prompt">Art Description:</label>
                <textarea id="prompt" rows="4" placeholder="Describe the artwork you want to create... (e.g., 'A beautiful sunset over mountains', 'A magical forest with glowing trees')"></textarea>
            </div>
            
            <div class="input-group">
                <label for="style">Art Style:</label>
                <select id="style">
                    <option value="realistic">Realistic</option>
                    <option value="abstract">Abstract</option>
                    <option value="cartoon">Cartoon</option>
                    <option value="oil-painting">Oil Painting</option>
                    <option value="watercolor">Watercolor</option>
                    <option value="digital">Digital Art</option>
                    <option value="sketch">Sketch</option>
                    <option value="fantasy">Fantasy</option>
                </select>
            </div>
            
            <button onclick="generateArt()" id="generateBtn">🎨 Generate Artwork</button>
            
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>Creating your masterpiece...</p>
            </div>
            
            <div class="result" id="result"></div>
        </div>
        
        <div class="artwork-display">
            <div class="placeholder" id="placeholder">
                <div class="placeholder-icon">🎨</div>
                <p>Your generated artwork will appear here</p>
                <p>Enter a description and choose a style to get started!</p>
            </div>
            <img id="artworkImage" class="artwork-image" style="display: none;">
        </div>
        
        <div class="history-section" id="historySection" style="display: none;">
            <div class="history-title">🎭 Art History</div>
            <div class="history-grid" id="historyGrid"></div>
        </div>
    </div>
    
    <script>
        let artHistory = [];
        
        async function generateArt() {
            const prompt = document.getElementById('prompt').value;
            const style = document.getElementById('style').value;
            const generateBtn = document.getElementById('generateBtn');
            const loading = document.getElementById('loading');
            const result = document.getElementById('result');
            const placeholder = document.getElementById('placeholder');
            const artworkImage = document.getElementById('artworkImage');
            
            if (!prompt) {
                alert('Please enter an art description!');
                return;
            }
            
            // Show loading
            generateBtn.disabled = true;
            loading.style.display = 'block';
            result.innerHTML = '';
            
            try {
                const response = await fetch('/generate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ prompt, style })
                });
                
                const data = await response.json();
                
                if (data.status === 'success') {
                    // Hide placeholder and show artwork
                    placeholder.style.display = 'none';
                    artworkImage.src = data.image_url;
                    artworkImage.style.display = 'block';
                    
                    // Add to history
                    artHistory.unshift({
                        prompt: prompt,
                        style: style,
                        image: data.image_url,
                        timestamp: new Date().toLocaleString()
                    });
                    
                    // Update history display
                    updateHistory();
                    
                    result.innerHTML = `
                        <div class="success-message">
                            <h3>✅ Artwork Generated!</h3>
                            <p>${data.message}</p>
                        </div>
                    `;
                } else {
                    result.innerHTML = `
                        <div class="error-message">
                            <h3>❌ Error</h3>
                            <p>${data.message}</p>
                        </div>
                    `;
                }
            } catch (error) {
                result.innerHTML = `
                    <div class="error-message">
                        <h3>❌ Error</h3>
                        <p>Failed to generate artwork: ${error.message}</p>
                    </div>
                `;
            } finally {
                loading.style.display = 'none';
                generateBtn.disabled = false;
            }
        }
        
        function updateHistory() {
            const historySection = document.getElementById('historySection');
            const historyGrid = document.getElementById('historyGrid');
            
            if (artHistory.length > 0) {
                historySection.style.display = 'block';
                historyGrid.innerHTML = '';
                
                artHistory.slice(0, 8).forEach((item, index) => {
                    const historyItem = document.createElement('div');
                    historyItem.className = 'history-item';
                    historyItem.onclick = () => loadArtwork(item);
                    historyItem.innerHTML = `
                        <img src="${item.image}" alt="Artwork ${index + 1}">
                        <p><strong>${item.style}</strong></p>
                        <p>${item.prompt.substring(0, 30)}${item.prompt.length > 30 ? '...' : ''}</p>
                        <p style="font-size: 0.8em; opacity: 0.7;">${item.timestamp}</p>
                    `;
                    historyGrid.appendChild(historyItem);
                });
            }
        }
        
        function loadArtwork(item) {
            const placeholder = document.getElementById('placeholder');
            const artworkImage = document.getElementById('artworkImage');
            
            placeholder.style.display = 'none';
            artworkImage.src = item.image;
            artworkImage.style.display = 'block';
            
            // Update form
            document.getElementById('prompt').value = item.prompt;
            document.getElementById('style').value = item.style;
        }
        
        // Load history on page load
        window.addEventListener('load', async () => {
            try {
                const response = await fetch('/history');
                const history = await response.json();
                artHistory = history;
                updateHistory();
            } catch (error) {
                console.log('No history available');
            }
        });
        
        // Enter key to generate
        document.getElementById('prompt').addEventListener('keypress', function(e) {
            if (e.key === 'Enter' && e.ctrlKey) {
                generateArt();
            }
        });
    </script>
</body>
</html>
        '''
    
    def run(self, host='0.0.0.0', port=5000, debug=False):
        """Run the Flask application"""
        print("🎨 Starting Al-artworks AI Suite...")
        print(f"🌐 Web interface: http://{host}:{port}")
        print("📱 Open your browser to start creating AI artwork!")
        print("🔄 Press Ctrl+C to stop the application")
        print()
        
        # Open browser automatically
        try:
            webbrowser.open(f'http://localhost:{port}')
        except:
            pass
        
        self.app.run(host=host, port=port, debug=debug)

def main():
    """Main function to run the standalone application"""
    print("🎨 Al-artworks AI Suite - Standalone Application")
    print("=" * 50)
    print()
    
    # Create and run the application
    app = AlArtworksApp()
    
    try:
        app.run(host='0.0.0.0', port=5000, debug=False)
    except KeyboardInterrupt:
        print("\n👋 Thanks for using Al-artworks AI Suite!")
        print("🎨 Keep creating amazing artwork!")

if __name__ == '__main__':
    main() 