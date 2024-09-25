import os
import subprocess
import importlib
import urllib.request

def download_file(url, output_path):
    """Download a file from the specified URL to the given output path."""
    try:
        print(f"Downloading from {url} to {output_path}...")
        urllib.request.urlretrieve(url, output_path)
        print(f"Downloaded {output_path} successfully.")
    except Exception as e:
        print(f"Error downloading {output_path}: {e}")
    return output_path

def is_package_installed(package_name):
    """Check if a package is installed."""
    try:
        importlib.import_module(package_name)
        print(f"{package_name} is already installed.")
        return True
    except ImportError:
        print(f"{package_name} is not installed.")
        return False

def install_package(package_name, version=None):
    """Install a package using pip."""
    try:
        if version:
            subprocess.run(["pip", "install", f"{package_name}=={version}"], check=True)
        else:
            subprocess.run(["pip", "install", package_name], check=True)
        print(f"{package_name} installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error installing {package_name}: {e}")

def clone_repository(repo_url, branch, folder_name):
    if not os.path.exists(folder_name):
        print(f"Cloning repository into {folder_name}...")
        subprocess.run(["git", "clone", "-b", branch, repo_url], check=True)
    else:
        print(f"Repository already exists at {folder_name}. Skipping clone.")

def setup_environment():
    try:
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        # Get the current working directory
        cd = os.getcwd()

        # Clone the repository
        clone_repository("https://github.com/camenduru/xtts2-hf", "dev", os.path.join(cd, "xtts2-hf"))

        # Navigate to the new directory
        new_directory = os.path.join(cd, "xtts2-hf")
        os.chdir(new_directory)

        # Install from requirements.txt if exists
        requirements_path = os.path.join(new_directory, "requirements.txt")
        if os.path.exists(requirements_path):
            print("Installing from requirements.txt...")
            subprocess.run(["pip", "install", "-r", "requirements.txt"], check=True)


        # Check and install required packages
        required_packages = {
            "blinker": None,
            "TTS": "0.21.1",
            "langid": None,
            "unidic_lite": None,
            "unidic": None,
            "deepspeed": None,
            "numpy": "<2.0.0",
            "SpeechRecognition": None,
            "googletrans": "4.0.0-rc1",
            "deep_translator": None,
            "flask_ngrok": None,
            "pyngrok": None
        }

        for package, version in required_packages.items():
            if not is_package_installed(package):
                install_package(package, version)

        # Download required files
        print("Downloading required files...")
        # Download required files using urllib instead of wget
        download_file("https://huggingface.co/spaces/coqui/xtts/resolve/main/examples/female.wav", "./examples/female.wav")
        download_file("https://huggingface.co/spaces/coqui/xtts/resolve/main/examples/male.wav", "./examples/male.wav")
        download_file("https://huggingface.co/spaces/coqui/xtts/resolve/main/ffmpeg.zip", "./ffmpeg.zip")


        print("Environment setup complete!")

    except subprocess.CalledProcessError as e:
        print(f"An error occurred during the setup: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
