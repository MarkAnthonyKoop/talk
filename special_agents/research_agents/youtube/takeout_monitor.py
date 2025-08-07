#!/usr/bin/env python3
"""
Takeout Monitor - Monitors for YouTube takeout changes and auto-reindexes

Features:
- Watches for new takeout files
- Automatically rebuilds database
- Re-categorizes with AI
- Notifies of significant changes
"""

import time
import hashlib
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
import logging

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    HAS_WATCHDOG = True
except ImportError:
    HAS_WATCHDOG = False
    print("Install watchdog for file monitoring: pip install watchdog")

from build_db_enhanced import build_enhanced_database
from category_tree_builder import CategoryTreeBuilder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TakeoutChangeHandler(FileSystemEventHandler):
    """Handles changes to takeout files."""
    
    def __init__(self, monitor):
        self.monitor = monitor
        
    def on_modified(self, event):
        if not event.is_directory:
            if 'takeout' in event.src_path.lower() or 'watch-history' in event.src_path.lower():
                logger.info(f"Detected change in: {event.src_path}")
                self.monitor.process_change(event.src_path)
    
    def on_created(self, event):
        if not event.is_directory:
            if 'takeout' in event.src_path.lower():
                logger.info(f"New takeout file detected: {event.src_path}")
                self.monitor.process_new_takeout(event.src_path)


class TakeoutMonitor:
    """Monitors and processes YouTube takeout data changes."""
    
    def __init__(self, watch_dir: Path = None, db_path: Path = None, auto_categorize: bool = True):
        """Initialize the monitor."""
        self.watch_dir = watch_dir or Path.home() / "Downloads"
        self.db_path = db_path or Path.home() / "code" / "miniapps" / "youtube_database" / "youtube_enhanced.db"
        self.auto_categorize = auto_categorize
        
        # State tracking
        self.state_file = Path.home() / ".youtube_monitor" / "state.json"
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state = self._load_state()
        
        # Category builder
        self.category_builder = None
        if self.db_path.exists():
            self.category_builder = CategoryTreeBuilder(str(self.db_path))
    
    def _load_state(self) -> Dict:
        """Load monitor state."""
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                return json.load(f)
        return {
            "last_processed": None,
            "last_hash": None,
            "video_count": 0,
            "processing_history": []
        }
    
    def _save_state(self):
        """Save monitor state."""
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)
    
    def process_change(self, file_path: str):
        """Process a change in takeout data."""
        file_path = Path(file_path)
        
        # Check if it's a significant change
        file_hash = self._get_file_hash(file_path)
        
        if file_hash == self.state.get("last_hash"):
            logger.info("No significant changes detected")
            return
        
        logger.info("Processing takeout changes...")
        
        # Rebuild database
        self._rebuild_database(file_path)
        
        # Re-categorize if enabled
        if self.auto_categorize and self.category_builder:
            self._recategorize()
        
        # Update state
        self.state["last_processed"] = datetime.now().isoformat()
        self.state["last_hash"] = file_hash
        self._save_state()
        
        # Notify of changes
        self._notify_changes()
    
    def process_new_takeout(self, file_path: str):
        """Process a new takeout file."""
        file_path = Path(file_path)
        
        if file_path.suffix == '.zip':
            logger.info(f"Processing new takeout: {file_path}")
            self.process_change(file_path)
    
    def _get_file_hash(self, file_path: Path) -> str:
        """Get hash of file for change detection."""
        hasher = hashlib.md5()
        
        with open(file_path, 'rb') as f:
            # Read in chunks for large files
            while chunk := f.read(8192):
                hasher.update(chunk)
        
        return hasher.hexdigest()
    
    def _rebuild_database(self, takeout_path: Path):
        """Rebuild the enhanced database."""
        logger.info("Rebuilding database...")
        
        try:
            # Get old video count
            old_count = self.state.get("video_count", 0)
            
            # Build new database
            build_enhanced_database(takeout_path, self.db_path)
            
            # Get new video count
            import sqlite3
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM videos")
            new_count = cursor.fetchone()[0]
            conn.close()
            
            # Track changes
            change = {
                "timestamp": datetime.now().isoformat(),
                "old_count": old_count,
                "new_count": new_count,
                "added": new_count - old_count
            }
            
            self.state["video_count"] = new_count
            self.state["processing_history"].append(change)
            
            logger.info(f"Database rebuilt: {new_count} videos ({new_count - old_count:+d} change)")
            
        except Exception as e:
            logger.error(f"Database rebuild failed: {e}")
    
    def _recategorize(self):
        """Re-run AI categorization."""
        logger.info("Re-categorizing videos...")
        
        try:
            # Rebuild default categories
            prompts = [
                "Create a hierarchical category tree focusing on educational value",
                "Organize by technical depth and complexity",
                "Group by practical application and project relevance"
            ]
            
            for i, prompt in enumerate(prompts):
                scheme_name = f"auto_{i}_{datetime.now().strftime('%Y%m%d')}"
                self.category_builder.build_category_tree(prompt, scheme_name)
                logger.info(f"Created scheme: {scheme_name}")
            
        except Exception as e:
            logger.error(f"Categorization failed: {e}")
    
    def _notify_changes(self):
        """Notify about significant changes."""
        recent = self.state["processing_history"][-1] if self.state["processing_history"] else None
        
        if not recent:
            return
        
        added = recent["added"]
        
        if added > 0:
            message = f"📺 New YouTube videos indexed: {added} videos added"
        elif added < 0:
            message = f"📺 YouTube videos removed: {abs(added)} videos"
        else:
            message = "📺 YouTube database updated (no count change)"
        
        # Log the notification
        logger.info(message)
        
        # Could also send desktop notification, email, etc.
        try:
            import subprocess
            if Path("/usr/bin/notify-send").exists():
                subprocess.run(["notify-send", "YouTube Monitor", message])
        except:
            pass
    
    def start_monitoring(self):
        """Start monitoring for changes."""
        if not HAS_WATCHDOG:
            logger.error("Watchdog not installed. Using polling mode.")
            self._polling_monitor()
            return
        
        logger.info(f"Starting monitor on: {self.watch_dir}")
        
        event_handler = TakeoutChangeHandler(self)
        observer = Observer()
        observer.schedule(event_handler, str(self.watch_dir), recursive=True)
        observer.start()
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            observer.stop()
            logger.info("Monitor stopped")
        
        observer.join()
    
    def _polling_monitor(self):
        """Fallback polling-based monitoring."""
        logger.info("Using polling mode (checking every 5 minutes)")
        
        known_files = {}
        
        while True:
            try:
                # Check for takeout files
                for file_path in self.watch_dir.glob("*takeout*.zip"):
                    file_hash = self._get_file_hash(file_path)
                    
                    if file_path not in known_files:
                        # New file
                        logger.info(f"Found new takeout: {file_path}")
                        self.process_new_takeout(file_path)
                        known_files[file_path] = file_hash
                        
                    elif known_files[file_path] != file_hash:
                        # Changed file
                        logger.info(f"Takeout changed: {file_path}")
                        self.process_change(file_path)
                        known_files[file_path] = file_hash
                
                time.sleep(300)  # Check every 5 minutes
                
            except KeyboardInterrupt:
                logger.info("Monitor stopped")
                break
            except Exception as e:
                logger.error(f"Polling error: {e}")
                time.sleep(60)
    
    def get_status(self) -> Dict:
        """Get current monitor status."""
        return {
            "watching": str(self.watch_dir),
            "database": str(self.db_path),
            "last_processed": self.state.get("last_processed"),
            "video_count": self.state.get("video_count", 0),
            "auto_categorize": self.auto_categorize,
            "history": self.state.get("processing_history", [])[-5:]  # Last 5 changes
        }


def main():
    """Run the takeout monitor."""
    import argparse
    
    parser = argparse.ArgumentParser(description="YouTube Takeout Monitor")
    parser.add_argument("--watch", default="~/Downloads", help="Directory to watch")
    parser.add_argument("--db", help="Database path")
    parser.add_argument("--no-categorize", action="store_true", help="Skip auto-categorization")
    parser.add_argument("--status", action="store_true", help="Show status and exit")
    
    args = parser.parse_args()
    
    watch_dir = Path(args.watch).expanduser()
    db_path = Path(args.db) if args.db else None
    
    monitor = TakeoutMonitor(
        watch_dir=watch_dir,
        db_path=db_path,
        auto_categorize=not args.no_categorize
    )
    
    if args.status:
        status = monitor.get_status()
        print("\n📊 YouTube Takeout Monitor Status")
        print("=" * 40)
        for key, value in status.items():
            if key != "history":
                print(f"{key}: {value}")
        
        if status["history"]:
            print("\nRecent processing history:")
            for entry in status["history"]:
                print(f"  {entry['timestamp']}: {entry['added']:+d} videos")
    else:
        print("🔍 YouTube Takeout Monitor")
        print(f"Watching: {watch_dir}")
        print(f"Database: {monitor.db_path}")
        print("Press Ctrl+C to stop\n")
        
        monitor.start_monitoring()


if __name__ == "__main__":
    main()