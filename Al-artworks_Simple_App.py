#!/usr/bin/env python3
"""
🎨 Al-artworks Simple AI Art Generator
=====================================

A lightweight standalone application for AI art generation.
No CMake, no complex dependencies - just pure Python art creation.
"""

import os
import sys
import base64
import io
import webbrowser
from datetime import datetime

# Auto-install dependencies
try:
    from flask import Flask, request, jsonify
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
except ImportError:
    print("📦 Installing required packages...")
    import subprocess
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", 
        "flask", "numpy", "matplotlib"
    ])
    from flask import Flask, request, jsonify
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


class SimpleArtApp:
    """Simple AI Art Generator - No CMake Required"""
    
    def __init__(self):
        self.app = Flask(__name__)
        self.setup_routes()
        self.art_history = []
        
    def setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def index():
            """Main page"""
            return self.get_html()
            
        @self.app.route('/generate', methods=['POST'])
        def generate_art():
            """Generate artwork"""
            try:
                data = request.get_json()
                prompt = data.get('prompt', '')
                style = data.get('style', 'realistic')
                
                if not prompt:
                    return jsonify({
                        'status': 'error', 
                        'message': 'Please provide an art description'
                    })
                
                # Generate artwork
                image_data = self.create_artwork(prompt, style)
                
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
                return jsonify({
                    'status': 'error', 
                    'message': str(e)
                }), 500
                
        @self.app.route('/history')
        def get_history():
            """Get art history"""
            return jsonify(self.art_history)
    
    def create_artwork(self, prompt, style):
        """Create artwork based on style"""
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        if style == 'realistic':
            self.draw_realistic(ax, prompt)
        elif style == 'abstract':
            self.draw_abstract(ax, prompt)
        elif style == 'cartoon':
            self.draw_cartoon(ax, prompt)
        elif style == 'fantasy':
            self.draw_fantasy(ax, prompt)
        else:
            self.draw_realistic(ax, prompt)
        
        # Convert to base64
        canvas = FigureCanvas(fig)
        canvas.draw()
        
        img_data = io.BytesIO()
        fig.savefig(img_data, format='png', dpi=150, bbox_inches='tight')
        img_data.seek(0)
        
        image_base64 = base64.b64encode(img_data.getvalue()).decode('utf-8')
        plt.close(fig)
        return image_base64
    
    def draw_realistic(self, ax, prompt):
        """Draw realistic artwork"""
        ax.clear()
        
        # Create landscape
        x = np.linspace(0, 10, 100)
        y = np.sin(x) * 2 + np.random.normal(0, 0.1, 100)
        
        # Sky
        ax.fill_between(x, 5, 8, alpha=0.6, color='skyblue')
        
        # Mountains
        mountains = np.sin(x * 0.5) * 3 + 2
        ax.fill_between(x, mountains, 8, alpha=0.8, color='gray')
        
        # Ground
        ax.fill_between(x, 0, mountains, alpha=0.7, color='green')
        
        # Trees
        tree_positions = np.random.choice(x, 15)
        for pos in tree_positions:
            ax.scatter(pos, np.sin(pos) * 2 + 1, s=100, 
                      color='darkgreen', marker='^')
        
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 8)
        ax.set_title(f'Realistic: {prompt}', fontsize=16, fontweight='bold')
        ax.axis('off')
    
    def draw_abstract(self, ax, prompt):
        """Draw abstract artwork"""
        ax.clear()
        
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
    
    def draw_cartoon(self, ax, prompt):
        """Draw cartoon artwork"""
        ax.clear()
        
        # Head
        head = patches.Circle((5, 5), 1.5, color='peachpuff', 
                             ec='black', linewidth=2)
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
        smile = patches.Arc((5, 4.8), 0.8, 0.4, angle=0, theta1=0, 
                           theta2=180, color='black', linewidth=2)
        ax.add_patch(smile)
        
        # Body
        body = patches.Rectangle((4.5, 2), 1, 2, color='lightblue', 
                                ec='black', linewidth=2)
        ax.add_patch(body)
        
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 8)
        ax.set_title(f'Cartoon: {prompt}', fontsize=16, fontweight='bold')
        ax.axis('off')
    
    def draw_fantasy(self, ax, prompt):
        """Draw fantasy artwork"""
        ax.clear()
        
        # Floating islands
        for i in range(5):
            island_x = np.random.rand() * 10
            island_y = np.random.rand() * 4 + 2
            island_size = np.random.rand() * 1 + 0.5
            
            island = patches.Circle((island_x, island_y), island_size, 
                                  color='green', alpha=0.8)
            ax.add_patch(island)
            
            # Trees
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
    
    def get_html(self):
        """Get HTML template"""
        return '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🎨 Al-artworks Simple AI Suite</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
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
            width: 100%;
            margin-top: 10px;
            transition: all 0.3s ease;
        }
        
        button:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.3);
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
        
        .result {
            margin-top: 30px;
            text-align: center;
        }
        
        .success-message {
            background: rgba(0,255,0,0.2);
            color: #ccffcc;
            padding: 15px;
            border-radius: 10px;
            margin-top: 15px;
            border: 1px solid rgba(0,255,0,0.3);
        }
        
        .error-message {
            background: rgba(255,0,0,0.2);
            color: #ffcccc;
            padding: 15px;
            border-radius: 10px;
            margin-top: 15px;
            border: 1px solid rgba(255,0,0,0.3);
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎨 Al-artworks Simple AI Suite</h1>
            <p>Create stunning AI-powered artwork - No CMake Required!</p>
        </div>
        
        <div class="art-generator">
            <div class="input-group">
                <label for="prompt">Art Description:</label>
                <textarea id="prompt" rows="4" placeholder="Describe the artwork you want to create..."></textarea>
            </div>
            
            <div class="input-group">
                <label for="style">Art Style:</label>
                <select id="style">
                    <option value="realistic">Realistic</option>
                    <option value="abstract">Abstract</option>
                    <option value="cartoon">Cartoon</option>
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
    </div>
    
    <script>
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
                    placeholder.style.display = 'none';
                    artworkImage.src = data.image_url;
                    artworkImage.style.display = 'block';
                    
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
        
        document.getElementById('prompt').addEventListener('keypress', function(e) {
            if (e.key === 'Enter' && e.ctrlKey) {
                generateArt();
            }
        });
    </script>
</body>
</html>
        '''
    
    def run(self, host='0.0.0.0', port=5000):
        """Run the application"""
        print("🎨 Starting Al-artworks Simple AI Suite...")
        print(f"🌐 Web interface: http://{host}:{port}")
        print("📱 Open your browser to start creating AI artwork!")
        print("🔄 Press Ctrl+C to stop the application")
        print()
        
        try:
            webbrowser.open(f'http://localhost:{port}')
        except:
            pass
        
        self.app.run(host=host, port=port, debug=False)


def main():
    """Main function"""
    print("🎨 Al-artworks Simple AI Suite - No CMake Required!")
    print("=" * 50)
    print()
    
    app = SimpleArtApp()
    
    try:
        app.run(host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n👋 Thanks for using Al-artworks Simple AI Suite!")
        print("🎨 Keep creating amazing artwork!")


if __name__ == '__main__':
    main() 