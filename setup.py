
"""
Automated Book Publication Workflow - Setup Script
Handles installation and initial configuration
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def run_command(command, description):
    """Run a shell command with error handling"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return None

def create_directories():
    """Create necessary directories"""
    directories = [
        "screenshots",
        "content",
        "logs",
        "data",
        "models",
        "exports"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"📁 Created directory: {directory}")

def install_dependencies():
    """Install Python dependencies"""
    print("📦 Installing Python dependencies...")
    
    # Upgrade pip first
    run_command(f"{sys.executable} -m pip install --upgrade pip", "Upgrading pip")
    
    # Install requirements
    run_command(f"{sys.executable} -m pip install -r requirements.txt", "Installing requirements")
    
    # Install Playwright browsers
    run_command("playwright install", "Installing Playwright browsers")

def setup_environment():
    """Setup environment variables"""
    env_template = ".env.template"
    env_file = ".env"
    
    if not os.path.exists(env_file) and os.path.exists(env_template):
        shutil.copy(env_template, env_file)
        print(f"📝 Created {env_file} from template")
        print("⚠️  Please update the API keys in .env file before running the application")
    else:
        print(f"✅ {env_file} already exists")

def setup_git_hooks():
    """Setup git hooks for development"""
    hooks_dir = Path(".git/hooks")
    if hooks_dir.exists():
        pre_commit_hook = hooks_dir / "pre-commit"
        with open(pre_commit_hook, "w") as f:
            f.write("""#!/bin/sh
# Pre-commit hook for code formatting and linting
echo "Running pre-commit checks..."

# Format code with black
black --check . || {
    echo "Code formatting required. Run: black ."
    exit 1
}

# Lint with flake8
flake8 . || {
    echo "Linting errors found. Fix them before committing."
    exit 1
}

echo "Pre-commit checks passed!"
""")
        pre_commit_hook.chmod(0o755)
        print("🔗 Git pre-commit hook installed")

def check_system_requirements():
    """Check system requirements"""
    print("🔍 Checking system requirements...")
    

    python_version = sys.version_info
    if python_version < (3, 8):
        print("❌ Python 3.8+ is required")
        return False
    else:
        print(f"✅ Python {python_version.major}.{python_version.minor} detected")
    

    required_tools = {
        'git': 'Git version control',
        'node': 'Node.js runtime',
        'npm': 'Node package manager'
    }
    
    missing_tools = []
    for tool, description in required_tools.items():
        if shutil.which(tool):
            print(f"✅ {description} found")
        else:
            print(f"⚠️  {description} not found - some features may not work")
            missing_tools.append(tool)
    
    return len(missing_tools) == 0

def create_config_files():
    """Create default configuration files"""
    configs = {
        "config.yaml": """# Book Publication Workflow Configuration
app:
  name: "Book Publication Workflow"
  version: "1.0.0"
  debug: false

database:
  path: "data/workflow.db"
  backup_interval: 3600  # seconds

publishing:
  platforms:
    - amazon_kdp
    - draft2digital
    - kobo
    - apple_books
  
  formats:
    - epub
    - mobi
    - pdf

automation:
  screenshot_interval: 30  # seconds
  max_retries: 3
  timeout: 60  # seconds

logging:
  level: "INFO"
  file: "logs/workflow.log"
  max_size: "10MB"
  backup_count: 5
""",
        "requirements.txt": """# Core dependencies
requests>=2.31.0
beautifulsoup4>=4.12.0
selenium>=4.15.0
playwright>=1.40.0
pillow>=10.0.0
pdfkit>=1.0.0
ebooklib>=0.18
python-dotenv>=1.0.0
pyyaml>=6.0
click>=8.1.0
rich>=13.0.0
schedule>=1.2.0
sqlalchemy>=2.0.0
jinja2>=3.1.0
watchdog>=3.0.0

# Development dependencies
black>=23.0.0
flake8>=6.0.0
pytest>=7.4.0
pytest-cov>=4.1.0
pre-commit>=3.5.0

# Optional AI/ML dependencies
openai>=1.0.0
anthropic>=0.7.0
""",
        ".env.template": """# API Keys (Copy this to .env and fill in your keys)
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Publishing Platform Credentials
AMAZON_KDP_EMAIL=your_email@example.com
AMAZON_KDP_PASSWORD=your_password_here

# Database Configuration
DATABASE_URL=sqlite:///data/workflow.db

# Application Settings
DEBUG=False
LOG_LEVEL=INFO

# Automation Settings
HEADLESS_BROWSER=True
SCREENSHOT_ENABLED=True
""",
        ".gitignore": """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Environment
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# Logs
logs/
*.log

# Data and exports
data/
exports/
screenshots/
models/

# OS
.DS_Store
Thumbs.db
"""
    }
    
    for filename, content in configs.items():
        if not os.path.exists(filename):
            with open(filename, 'w') as f:
                f.write(content)
            print(f"📄 Created {filename}")
        else:
            print(f"✅ {filename} already exists")

def setup_database():
    """Initialize the database"""
    print("🗄️  Setting up database...")
    

    Path("data").mkdir(exist_ok=True)
    

    db_init_script = """
import sqlite3
from pathlib import Path

def init_database():
    db_path = Path("data/workflow.db")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Books table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS books (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            author TEXT,
            genre TEXT,
            status TEXT DEFAULT 'draft',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Publishing sessions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS publishing_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            book_id INTEGER,
            platform TEXT NOT NULL,
            status TEXT DEFAULT 'pending',
            started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP,
            error_message TEXT,
            FOREIGN KEY (book_id) REFERENCES books (id)
        )
    ''')
    
    # Screenshots table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS screenshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER,
            filename TEXT NOT NULL,
            step_description TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES publishing_sessions (id)
        )
    ''')
    
    conn.commit()
    conn.close()
    print("✅ Database initialized successfully")

if __name__ == "__main__":
    init_database()
"""
    
    with open("init_db.py", "w") as f:
        f.write(db_init_script)
    

    run_command(f"{sys.executable} init_db.py", "Initializing database")

def create_sample_workflow():
    """Create a sample workflow script"""
    sample_workflow = """#!/usr/bin/env python3
'''
Sample Book Publishing Workflow
This demonstrates the basic workflow structure
'''
import os
import sys
import time
from pathlib import Path
from datetime import datetime

class BookPublisher:
    def __init__(self, book_title, platforms=None):
        self.book_title = book_title
        self.platforms = platforms or ['amazon_kdp']
        self.session_id = None
        self.screenshots_dir = Path("screenshots")
        
    def setup_session(self):
        '''Initialize a new publishing session'''
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_id = f"{self.book_title.replace(' ', '_')}_{timestamp}"
        session_dir = self.screenshots_dir / self.session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        print(f"📚 Starting publishing session: {self.session_id}")
        
    def take_screenshot(self, step_name):
        '''Take a screenshot of current step'''
        if not self.session_id:
            return
            
        timestamp = datetime.now().strftime("%H%M%S")
        filename = f"{step_name}_{timestamp}.png"
        # Placeholder for actual screenshot logic
        print(f"📸 Screenshot: {filename}")
        
    def publish_to_platform(self, platform):
        '''Publish book to specific platform'''
        print(f"🚀 Publishing '{self.book_title}' to {platform}")
        
        steps = [
            "login",
            "create_book",
            "upload_content",
            "set_metadata",
            "configure_pricing",
            "submit_for_review"
        ]
        
        for step in steps:
            print(f"   • {step.replace('_', ' ').title()}")
            self.take_screenshot(f"{platform}_{step}")
            time.sleep(2)  # Simulate processing time
            
        print(f"✅ Successfully published to {platform}")
        
    def run(self):
        '''Execute the complete publishing workflow'''
        self.setup_session()
        
        for platform in self.platforms:
            try:
                self.publish_to_platform(platform)
            except Exception as e:
                print(f"❌ Failed to publish to {platform}: {e}")
                
        print(f"🎉 Publishing workflow completed for '{self.book_title}'")

def main():
    '''Main entry point'''
    if len(sys.argv) < 2:
        print("Usage: python sample_workflow.py <book_title>")
        return
        
    book_title = " ".join(sys.argv[1:])
    publisher = BookPublisher(book_title)
    publisher.run()

if __name__ == "__main__":
    main()
"""
    
    with open("sample_workflow.py", "w") as f:
        f.write(sample_workflow)
    print("📖 Created sample_workflow.py")

def display_next_steps():
    """Display next steps for the user"""
    print("\n" + "="*60)
    print("🎉 SETUP COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\n📋 Next Steps:")
    print("1. Update API keys in .env file")
    print("2. Configure settings in config.yaml")
    print("3. Test the setup: python sample_workflow.py 'My Test Book'")
    print("4. Start building your automation scripts")
    print("\n📚 Useful Commands:")
    print("• Run sample workflow: python sample_workflow.py 'Book Title'")
    print("• View logs: tail -f logs/workflow.log")
    print("• Reset database: rm data/workflow.db && python init_db.py")
    print("\n🔧 Development:")
    print("• Format code: black .")
    print("• Run linting: flake8 .")
    print("• Run tests: pytest")
    print("\n" + "="*60)

def main():
    """Main setup function"""
    print("🚀 Book Publication Workflow - Setup")
    print("="*50)
    

    if not check_system_requirements():
        print("⚠️  Some system requirements are missing, but continuing with setup...")
    

    create_directories()
    

    create_config_files()
    

    install_dependencies()
    

    setup_environment()
    

    setup_database()
    

    setup_git_hooks()
    

    create_sample_workflow()
    

    display_next_steps()

if __name__ == "__main__":
    main()