#!/usr/bin/env python3
"""
Smart Dependencies Installation Script
Handles complex dependency installation with proper order and error handling
"""

import subprocess
import sys
import os
from pathlib import Path

def run_pip_install(packages, description="Installing packages"):
    """Install packages with proper error handling"""
    print(f"🔄 {description}...")
    
    if isinstance(packages, str):
        packages = [packages]
    
    for package in packages:
        try:
            cmd = [sys.executable, "-m", "pip", "install", package]
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"✅ Successfully installed: {package}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {e}")
            print(f"Error output: {e.stderr}")
            return False
    return True

def install_core_dependencies():
    """Install core Python dependencies first"""
    core_packages = [
        "pip>=23.0",
        "setuptools>=68.0",
        "wheel>=0.41.0",
        "numpy>=1.24.0", 
        "python-dotenv>=1.0.0",
        "pyyaml>=6.0.1",
        "requests>=2.31.0",
        "click>=8.1.0",
    ]
    
    return run_pip_install(core_packages, "Installing core dependencies")

def install_web_automation():
    """Install web automation libraries"""
    web_packages = [
        "selenium>=4.15.0",
        "beautifulsoup4>=4.12.0",
        "lxml>=4.9.0",
        "pillow>=10.0.0",
        "playwright>=1.40.0",
    ]
    
    success = run_pip_install(web_packages, "Installing web automation libraries")
    
    if success:

        try:
            print("🔄 Installing Playwright browsers...")
            subprocess.run([sys.executable, "-m", "playwright", "install"], check=True)
            print("✅ Playwright browsers installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install Playwright browsers: {e}")
            success = False
    
    return success

def install_ai_ml_libraries():
    """Install AI/ML libraries"""

    ml_base = [
        "scikit-learn>=1.3.0",
        "pandas>=2.0.0",
        "scipy>=1.11.0",
    ]
    
    if not run_pip_install(ml_base, "Installing ML base libraries"):
        return False
    

    embedding_packages = [
        "sentence-transformers>=2.2.2",
        "chromadb>=0.4.15",
    ]
    
    if not run_pip_install(embedding_packages, "Installing embedding libraries"):
        return False
    

    ai_packages = [
        "google-generativeai>=0.3.0",
        "openai>=1.3.0",
        "anthropic>=0.7.0",
    ]
    
    return run_pip_install(ai_packages, "Installing AI API libraries")

def install_ui_framework():
    """Install Gradio UI framework"""
    ui_packages = [
        "gradio>=4.8.0",
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
    ]
    
    return run_pip_install(ui_packages, "Installing UI framework")

def install_document_processing():
    """Install document processing libraries"""
    doc_packages = [
        "pdfkit>=1.0.0",
        "ebooklib>=0.18",
        "python-docx>=0.8.11",
        "openpyxl>=3.1.0",
        "reportlab>=4.0.0",
        "markdown>=3.5.0",
    ]
    
    return run_pip_install(doc_packages, "Installing document processing libraries")

def install_async_libraries():
    """Install async and HTTP libraries"""
    async_packages = [
        "aiohttp>=3.9.0",
        "aiofiles>=23.2.0",
        "httpx>=0.25.0",
    ]
    
    return run_pip_install(async_packages, "Installing async libraries")

def install_database_libraries():
    """Install database libraries"""
    db_packages = [
        "sqlalchemy>=2.0.0",
        "pydantic>=2.5.0",
    ]
    
    return run_pip_install(db_packages, "Installing database libraries")

def install_utility_libraries():
    """Install utility libraries"""
    util_packages = [
        "rich>=13.7.0",
        "loguru>=0.7.0",
        "schedule>=1.2.0",
        "watchdog>=3.0.0",
        "jinja2>=3.1.0",
        "python-dateutil>=2.8.0",
        "psutil>=5.9.0",
        "tqdm>=4.66.0",
        "cryptography>=41.0.0",
    ]
    
    return run_pip_install(util_packages, "Installing utility libraries")

def install_development_tools():
    """Install development and testing tools"""
    dev_packages = [
        "black>=23.11.0",
        "flake8>=6.1.0",
        "pytest>=7.4.0",
        "pytest-cov>=4.1.0",
        "pytest-asyncio>=0.21.0",
        "pre-commit>=3.5.0",
        "mypy>=1.7.0",
    ]
    
    return run_pip_install(dev_packages, "Installing development tools")

def install_optional_libraries():
    """Install optional libraries with error tolerance"""
    optional_packages = [
        "opencv-python>=4.8.0",
        "nltk>=3.8.0", 

        "celery>=5.3.0", 
    ]
    
    print("🔄 Installing optional libraries (failures are acceptable)...")
    for package in optional_packages:
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", package], 
                         check=True, capture_output=True, text=True)
            print(f"✅ Optional package installed: {package}")
        except subprocess.CalledProcessError:
            print(f"⚠️  Optional package skipped: {package}")
    
    return True

def create_requirements_txt():
    """Create requirements.txt file"""
    requirements_content = """# Book Publication Workflow - Core Requirements
# Generated by smart installer

# Core dependencies
numpy>=1.24.0
python-dotenv>=1.0.0
pyyaml>=6.0.1
requests>=2.31.0
click>=8.1.0

# Web automation
selenium>=4.15.0
playwright>=1.40.0
beautifulsoup4>=4.12.0
lxml>=4.9.0
pillow>=10.0.0

# AI/ML
google-generativeai>=0.3.0
openai>=1.3.0
anthropic>=0.7.0
chromadb>=0.4.15
sentence-transformers>=2.2.2
scikit-learn>=1.3.0
pandas>=2.0.0

# UI
gradio>=4.8.0

# Database
sqlalchemy>=2.0.0
pydantic>=2.5.0

# Document processing
pdfkit>=1.0.0
ebooklib>=0.18
python-docx>=0.8.11
reportlab>=4.0.0

# Async
aiohttp>=3.9.0
aiofiles>=23.2.0
httpx>=0.25.0

# Utilities
rich>=13.7.0
loguru>=0.7.0
schedule>=1.2.0
watchdog>=3.0.0
jinja2>=3.1.0
tqdm>=4.66.0

# Development
black>=23.11.0
flake8>=6.1.0
pytest>=7.4.0
pytest-asyncio>=0.21.0
"""
    
    with open("requirements.txt", "w") as f:
        f.write(requirements_content)
    print("📄 Created requirements.txt")

def setup_environment_files():
    """Create necessary environment files"""
    

    env_template = """# Book Publication Workflow - Environment Variables
# Copy this file to .env and fill in your actual values

# AI API Keys
GEMINI_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Application Settings
DEBUG=False
LOG_LEVEL=INFO

# Database
DATABASE_URL=sqlite:///data/workflow.db

# Web Automation
HEADLESS_BROWSER=True
SCREENSHOT_ENABLED=True
BROWSER_TIMEOUT=30

# Publishing Platforms (if using automation)
AMAZON_KDP_EMAIL=your_email@example.com
AMAZON_KDP_PASSWORD=your_secure_password

# Optional: Notification Settings
# SMTP_SERVER=smtp.gmail.com
# SMTP_PORT=587
# EMAIL_USER=your_email@gmail.com
# EMAIL_PASSWORD=your_app_password
"""
    
    with open(".env.template", "w") as f:
        f.write(env_template)
    print("📄 Created .env.template")

def main():
    """Main installation function"""
    print("🚀 Book Publication Workflow - Smart Installer")
    print("="*60)
    

    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required")
        sys.exit(1)
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    

    directories = ["screenshots", "content", "logs", "data", "models", "exports"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"📁 Created directory: {directory}")
    

    installation_steps = [
        ("Core Dependencies", install_core_dependencies),
        ("Web Automation", install_web_automation),
        ("AI/ML Libraries", install_ai_ml_libraries),
        ("UI Framework", install_ui_framework),
        ("Document Processing", install_document_processing),
        ("Async Libraries", install_async_libraries),
        ("Database Libraries", install_database_libraries),
        ("Utility Libraries", install_utility_libraries),
        ("Development Tools", install_development_tools),
        ("Optional Libraries", install_optional_libraries),
    ]
    
    failed_steps = []
    for step_name, step_function in installation_steps:
        print(f"\n📦 {step_name}")
        print("-" * 40)
        if not step_function():
            failed_steps.append(step_name)
            print(f"⚠️  {step_name} had some failures")
        else:
            print(f"✅ {step_name} completed successfully")
    

    create_requirements_txt()
    setup_environment_files()
    

    print("\n" + "="*60)
    print("🎉 INSTALLATION COMPLETED!")
    print("="*60)
    
    if failed_steps:
        print(f"⚠️  Some steps had failures: {', '.join(failed_steps)}")
        print("You may need to install these manually or check system requirements.")
    
    print("\n📋 Next Steps:")
    print("1. Copy .env.template to .env and add your API keys")
    print("2. Test the installation:")
    print("   python -c 'import gradio, playwright, chromadb; print(\"✅ All core libraries working!\")'")
    print("3. Run the main application:")
    print("   python your_main_application.py")
    
    print("\n🔧 Troubleshooting:")
    print("• If ChromaDB fails: pip install --force-reinstall chromadb")
    print("• If Playwright fails: playwright install --with-deps")
    print("• If AI libraries fail: Check your Python version and OS compatibility")

if __name__ == "__main__":
    main()