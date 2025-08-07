import requests
import os
import logging
import json
import hashlib
import shutil
import subprocess
import sys
from typing import Optional, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger("UpdateChecker")

@dataclass
class ReleaseInfo:
    """Information about a GitHub release"""
    version: str
    name: str
    body: str
    published_at: str
    assets: list
    download_url: Optional[str] = None
    file_size: Optional[int] = None
    checksum: Optional[str] = None

class UpdateChecker:
    """GitHub release checker with automatic update capabilities"""
    
    def __init__(self, repo_owner: str, repo_name: str, current_version: str = "1.0.0"):
        self.repo_owner = repo_owner
        self.repo_name = repo_name
        self.current_version = current_version
        self.github_api_base = "https://api.github.com"
        self.local_version_file = "version.json"
        self.download_dir = "updates"
        self.backup_dir = "backups"
        
        # Ensure directories exist
        os.makedirs(self.download_dir, exist_ok=True)
        os.makedirs(self.backup_dir, exist_ok=True)

    def get_local_version(self) -> str:
        """Get local version from version file"""
        try:
            if os.path.exists(self.local_version_file):
                with open(self.local_version_file, 'r') as f:
                    data = json.load(f)
                    return data.get("version", self.current_version)
        except Exception as e:
            logger.error(f"Failed to read local version: {e}")
        
        return self.current_version

    def save_local_version(self, version: str):
        """Save local version to file"""
        try:
            version_data = {
                "version": version,
                "updated_at": self._get_current_time(),
                "update_source": "github"
            }
            with open(self.local_version_file, 'w') as f:
                json.dump(version_data, f, indent=2)
            logger.info(f"Local version saved: {version}")
        except Exception as e:
            logger.error(f"Failed to save local version: {e}")

    def _get_current_time(self) -> str:
        """Get current timestamp"""
        import datetime
        return datetime.datetime.now().isoformat()

    def _get_latest_release(self) -> Optional[ReleaseInfo]:
        """Get latest release information from GitHub"""
        try:
            url = f"{self.github_api_base}/repos/{self.repo_owner}/{self.repo_name}/releases/latest"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            release_data = response.json()
            
            # Find executable asset
            executable_asset = None
            for asset in release_data.get("assets", []):
                asset_name = asset["name"].lower()
                if any(ext in asset_name for ext in [".exe", ".dmg", ".deb", ".rpm", ".AppImage"]):
                    executable_asset = asset
                    break
            
            if executable_asset:
                return ReleaseInfo(
                    version=release_data["tag_name"],
                    name=release_data["name"],
                    body=release_data["body"],
                    published_at=release_data["published_at"],
                    assets=release_data["assets"],
                    download_url=executable_asset["browser_download_url"],
                    file_size=executable_asset["size"]
                )
            else:
                logger.warning("No executable asset found in latest release")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch latest release: {e}")
            return None
        except Exception as e:
            logger.error(f"Error parsing release data: {e}")
            return None

    def _download_file(self, url: str, filename: str) -> bool:
        """Download file from URL"""
        try:
            logger.info(f"Downloading {filename} from {url}")
            
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            file_path = os.path.join(self.download_dir, filename)
            
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            logger.info(f"Download completed: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download {filename}: {e}")
            return False

    def _verify_download(self, file_path: str, expected_size: Optional[int] = None) -> bool:
        """Verify downloaded file"""
        try:
            if not os.path.exists(file_path):
                logger.error(f"Downloaded file not found: {file_path}")
                return False
            
            actual_size = os.path.getsize(file_path)
            logger.info(f"Downloaded file size: {actual_size} bytes")
            
            if expected_size and actual_size != expected_size:
                logger.error(f"File size mismatch: expected {expected_size}, got {actual_size}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to verify download: {e}")
            return False

    def _create_backup(self) -> bool:
        """Create backup of current installation"""
        try:
            backup_name = f"backup_{self._get_current_time().replace(':', '-')}"
            backup_path = os.path.join(self.backup_dir, backup_name)
            
            # Copy current executable and important files
            current_exe = sys.executable
            if os.path.exists(current_exe):
                shutil.copy2(current_exe, backup_path)
                logger.info(f"Backup created: {backup_path}")
                return True
            else:
                logger.warning("Could not find current executable for backup")
                return False
                
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            return False

    def _install_update(self, file_path: str) -> bool:
        """Install the downloaded update"""
        try:
            logger.info(f"Installing update: {file_path}")
            
            # Create backup first
            if not self._create_backup():
                logger.warning("Failed to create backup, proceeding anyway")
            
            # For Windows, we might need to stop the current process
            if platform.system() == "Windows":
                # Schedule the update for next restart
                update_script = self._create_update_script(file_path)
                subprocess.run(["schtasks", "/create", "/tn", "ChatGPTPlusUpdate", 
                              "/tr", f'"{update_script}"', "/sc", "onstart"], 
                             shell=True, check=True)
                logger.info("Update scheduled for next system restart")
                return True
            else:
                # For Unix-like systems, try to replace directly
                current_exe = sys.executable
                shutil.copy2(file_path, current_exe)
                os.chmod(current_exe, 0o755)  # Make executable
                logger.info("Update installed successfully")
                return True
                
        except Exception as e:
            logger.error(f"Failed to install update: {e}")
            return False

    def _create_update_script(self, file_path: str) -> str:
        """Create update script for Windows"""
        script_content = f'''@echo off
timeout /t 5 /nobreak > nul
copy "{file_path}" "{sys.executable}"
del "{file_path}"
schtasks /delete /tn ChatGPTPlusUpdate /f
start "" "{sys.executable}"
'''
        script_path = os.path.join(self.download_dir, "update.bat")
        with open(script_path, 'w') as f:
            f.write(script_content)
        return script_path

    def check_for_updates(self) -> Optional[ReleaseInfo]:
        """Check for available updates"""
        try:
            local_version = self.get_local_version()
            logger.info(f"Local version: {local_version}")
            
            latest_release = self._get_latest_release()
            if not latest_release:
                logger.warning("Could not fetch latest release information")
                return None
            
            logger.info(f"Latest version: {latest_release.version}")
            
            if self._is_newer_version(latest_release.version, local_version):
                logger.info(f"Update available: {local_version} -> {latest_release.version}")
                return latest_release
            else:
                logger.info("No updates available")
                return None
                
        except Exception as e:
            logger.error(f"Error checking for updates: {e}")
            return None

    def _is_newer_version(self, new_version: str, current_version: str) -> bool:
        """Compare version strings to determine if new version is newer"""
        try:
            from packaging import version
            return version.parse(new_version) > version.parse(current_version)
        except ImportError:
            # Fallback to simple string comparison
            return new_version > current_version

    def download_and_install_update(self, release_info: ReleaseInfo) -> bool:
        """Download and install the update"""
        try:
            if not release_info.download_url:
                logger.error("No download URL available")
                return False
            
            # Extract filename from URL
            filename = release_info.download_url.split('/')[-1]
            
            # Download the file
            if not self._download_file(release_info.download_url, filename):
                return False
            
            file_path = os.path.join(self.download_dir, filename)
            
            # Verify download
            if not self._verify_download(file_path, release_info.file_size):
                return False
            
            # Install update
            if self._install_update(file_path):
                # Save new version
                self.save_local_version(release_info.version)
                logger.info(f"Update completed successfully to version {release_info.version}")
                return True
            else:
                return False
                
        except Exception as e:
            logger.error(f"Error during update process: {e}")
            return False

    def auto_update(self) -> bool:
        """Automatically check for and install updates"""
        try:
            logger.info("Checking for updates...")
            
            release_info = self.check_for_updates()
            if release_info:
                logger.info(f"Update found: {release_info.version}")
                logger.info(f"Release notes: {release_info.body[:200]}...")
                
                # Ask for confirmation (in a real app, this would be a UI dialog)
                logger.info("Update available. Installing...")
                
                return self.download_and_install_update(release_info)
            else:
                logger.info("No updates available")
                return False
                
        except Exception as e:
            logger.error(f"Auto update failed: {e}")
            return False

    def get_update_info(self) -> Dict[str, Any]:
        """Get update information"""
        try:
            local_version = self.get_local_version()
            latest_release = self._get_latest_release()
            
            info = {
                "local_version": local_version,
                "update_available": False,
                "latest_version": None,
                "release_notes": None,
                "download_size": None
            }
            
            if latest_release:
                info["latest_version"] = latest_release.version
                info["update_available"] = self._is_newer_version(latest_release.version, local_version)
                info["release_notes"] = latest_release.body
                info["download_size"] = latest_release.file_size
            
            return info
            
        except Exception as e:
            logger.error(f"Failed to get update info: {e}")
            return {"error": str(e)}

# Example usage
def example_update_checker():
    """Example of using the update checker"""
    logging.basicConfig(level=logging.INFO)
    
    # Create update checker (replace with your repo details)
    checker = UpdateChecker(
        repo_owner="yourusername",
        repo_name="chatgpt-plus-clone",
        current_version="1.0.0"
    )
    
    # Check for updates
    update_info = checker.get_update_info()
    logger.info(f"Update info: {update_info}")
    
    # Auto update
    if update_info.get("update_available", False):
        logger.info("Update available, installing...")
        success = checker.auto_update()
        if success:
            logger.info("Update completed successfully")
        else:
            logger.error("Update failed")
    else:
        logger.info("No updates available")

if __name__ == "__main__":
    import platform
    example_update_checker()