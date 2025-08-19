import subprocess
import sys

def run_streamlit():
    """Run the Streamlit application"""
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running Streamlit: {e}")
    except KeyboardInterrupt:
        print("\nApplication stopped by user")

if __name__ == "__main__":
    run_streamlit()
