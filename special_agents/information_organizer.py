"""
InformationOrganizer - Categorizes and organizes incoming information.

This module automatically categorizes information from various sources,
maintains a knowledge graph, and provides intelligent retrieval.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Set, Tuple
from collections import defaultdict, Counter
from pathlib import Path
import re

log = logging.getLogger(__name__)


class Category:
    """Represents a category of information."""
    
    def __init__(self, name: str, keywords: List[str], priority: int = 5):
        """Initialize a category."""
        self.name = name
        self.keywords = set(keywords)
        self.priority = priority
        self.item_count = 0
        self.last_updated = None
        self.subcategories = {}
        self.patterns = []  # Regex patterns for matching
        
    def matches(self, text: str) -> float:
        """
        Calculate match score for text against this category.
        
        Returns:
            Score from 0 to 1 indicating match strength
        """
        text_lower = text.lower()
        
        # Keyword matching
        keyword_matches = sum(1 for kw in self.keywords if kw in text_lower)
        keyword_score = keyword_matches / max(len(self.keywords), 1)
        
        # Pattern matching
        pattern_score = 0
        if self.patterns:
            pattern_matches = sum(1 for p in self.patterns if re.search(p, text_lower))
            pattern_score = pattern_matches / len(self.patterns)
        
        # Combined score
        return max(keyword_score, pattern_score)
    
    def add_subcategory(self, name: str, keywords: List[str]):
        """Add a subcategory."""
        self.subcategories[name] = Category(name, keywords, self.priority - 1)


class InformationItem:
    """Represents a piece of organized information."""
    
    def __init__(self,
                 content: Any,
                 source: str,
                 category: str,
                 timestamp: Optional[datetime] = None,
                 metadata: Optional[Dict] = None):
        """Initialize an information item."""
        self.id = f"{source}_{datetime.now().timestamp()}"
        self.content = content
        self.source = source
        self.category = category
        self.subcategories = []
        self.timestamp = timestamp or datetime.now()
        self.metadata = metadata or {}
        self.tags = set()
        self.relevance_score = 0
        self.access_count = 0
        self.last_accessed = None
        self.related_items = []  # IDs of related items
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "content": self.content,
            "source": self.source,
            "category": self.category,
            "subcategories": self.subcategories,
            "timestamp": self.timestamp.isoformat(),
            "tags": list(self.tags),
            "relevance_score": self.relevance_score,
            "access_count": self.access_count,
            "metadata": self.metadata
        }


class InformationOrganizer:
    """
    Organizes and categorizes information from multiple sources.
    
    This class automatically categorizes incoming information,
    maintains relationships between items, and provides intelligent
    retrieval based on context and relevance.
    """
    
    def __init__(self, save_path: Optional[Path] = None):
        """
        Initialize the information organizer.
        
        Args:
            save_path: Path to save organized information
        """
        self.save_path = save_path or Path(".talk_scratch/organized_info")
        self.save_path.mkdir(parents=True, exist_ok=True)
        
        # Categories
        self.categories = self._initialize_default_categories()
        self.custom_categories = {}
        
        # Information storage
        self.items: Dict[str, InformationItem] = {}
        self.category_index = defaultdict(list)  # category -> item IDs
        self.source_index = defaultdict(list)    # source -> item IDs
        self.tag_index = defaultdict(list)       # tag -> item IDs
        self.time_index = []  # Sorted list of (timestamp, item_id)
        
        # Statistics
        self.stats = {
            "total_items": 0,
            "items_by_category": Counter(),
            "items_by_source": Counter(),
            "recent_categories": deque(maxlen=10)
        }
        
        # Learning
        self.category_patterns = defaultdict(list)  # Learn patterns over time
        self.cooccurrence_matrix = defaultdict(lambda: defaultdict(int))
        
        log.info("Initialized InformationOrganizer")
    
    def _initialize_default_categories(self) -> Dict[str, Category]:
        """Initialize default categories."""
        categories = {
            "personal": Category("personal", [
                "i", "me", "my", "myself", "personal", "private", 
                "family", "home", "life", "feel", "think"
            ], priority=8),
            
            "work": Category("work", [
                "work", "job", "project", "task", "meeting", "deadline",
                "office", "colleague", "boss", "client", "business"
            ], priority=7),
            
            "technical": Category("technical", [
                "code", "programming", "software", "api", "database",
                "server", "bug", "feature", "deploy", "test", "debug"
            ], priority=6),
            
            "communication": Category("communication", [
                "email", "message", "call", "chat", "reply", "send",
                "contact", "reach", "discuss", "conversation"
            ], priority=5),
            
            "scheduling": Category("scheduling", [
                "time", "date", "when", "schedule", "calendar", "appointment",
                "tomorrow", "today", "week", "month", "reminder"
            ], priority=6),
            
            "financial": Category("financial", [
                "money", "pay", "cost", "price", "budget", "expense",
                "invoice", "bill", "salary", "investment", "bank"
            ], priority=7),
            
            "health": Category("health", [
                "health", "doctor", "medicine", "sick", "hospital",
                "exercise", "diet", "sleep", "wellness", "appointment"
            ], priority=8),
            
            "learning": Category("learning", [
                "learn", "study", "course", "tutorial", "documentation",
                "understand", "knowledge", "skill", "practice", "research"
            ], priority=5),
            
            "social": Category("social", [
                "friend", "party", "event", "meet", "social", "gathering",
                "dinner", "lunch", "coffee", "drinks", "weekend"
            ], priority=4),
            
            "general": Category("general", [""], priority=1)
        }
        
        # Add subcategories
        categories["work"].add_subcategory("meetings", ["meeting", "standup", "review", "presentation"])
        categories["work"].add_subcategory("projects", ["project", "milestone", "deliverable", "sprint"])
        categories["technical"].add_subcategory("development", ["code", "implement", "develop", "build"])
        categories["technical"].add_subcategory("debugging", ["bug", "fix", "error", "issue", "problem"])
        
        return categories
    
    def add_custom_category(self, name: str, keywords: List[str], priority: int = 5):
        """Add a custom category."""
        self.custom_categories[name] = Category(name, keywords, priority)
        log.info(f"Added custom category: {name}")
    
    def categorize(self, text: str, source: str = "unknown") -> Tuple[str, float]:
        """
        Categorize a piece of text.
        
        Args:
            text: The text to categorize
            source: Source of the text
            
        Returns:
            Tuple of (category_name, confidence_score)
        """
        best_category = "general"
        best_score = 0
        
        # Check all categories (default + custom)
        all_categories = {**self.categories, **self.custom_categories}
        
        for cat_name, category in all_categories.items():
            score = category.matches(text)
            
            # Boost score based on source patterns
            if source in self.category_patterns[cat_name]:
                score *= 1.2
            
            if score > best_score:
                best_score = score
                best_category = cat_name
        
        # Learn from this categorization
        if best_score > 0.5:
            self.category_patterns[best_category].append(source)
            self.stats["recent_categories"].append(best_category)
        
        return best_category, best_score
    
    def organize(self,
                 content: Any,
                 source: str,
                 metadata: Optional[Dict] = None,
                 auto_categorize: bool = True) -> InformationItem:
        """
        Organize a piece of information.
        
        Args:
            content: The information content
            source: Source of the information
            metadata: Additional metadata
            auto_categorize: Whether to automatically categorize
            
        Returns:
            The organized InformationItem
        """
        # Auto-categorize if requested
        if auto_categorize:
            text = str(content) if not isinstance(content, str) else content
            category, confidence = self.categorize(text, source)
        else:
            category = "general"
            confidence = 0
        
        # Create item
        item = InformationItem(
            content=content,
            source=source,
            category=category,
            metadata=metadata
        )
        item.relevance_score = confidence
        
        # Extract tags
        item.tags = self._extract_tags(str(content))
        
        # Find related items
        item.related_items = self._find_related_items(item)
        
        # Store item
        self.items[item.id] = item
        
        # Update indices
        self.category_index[category].append(item.id)
        self.source_index[source].append(item.id)
        for tag in item.tags:
            self.tag_index[tag].append(item.id)
        self.time_index.append((item.timestamp, item.id))
        self.time_index.sort(key=lambda x: x[0])
        
        # Update statistics
        self.stats["total_items"] += 1
        self.stats["items_by_category"][category] += 1
        self.stats["items_by_source"][source] += 1
        
        # Update co-occurrence
        for other_cat in self.stats["recent_categories"]:
            if other_cat != category:
                self.cooccurrence_matrix[category][other_cat] += 1
        
        # Auto-save periodically
        if self.stats["total_items"] % 50 == 0:
            self.save_state()
        
        log.debug(f"Organized item: {item.id} -> {category} (confidence: {confidence:.2f})")
        
        return item
    
    def _extract_tags(self, text: str) -> Set[str]:
        """Extract tags from text."""
        tags = set()
        
        # Extract hashtags
        hashtags = re.findall(r'#\w+', text)
        tags.update(hashtags)
        
        # Extract @mentions
        mentions = re.findall(r'@\w+', text)
        tags.update(mentions)
        
        # Extract URLs
        urls = re.findall(r'https?://\S+', text)
        if urls:
            tags.add("#has_url")
        
        # Extract emails
        emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
        if emails:
            tags.add("#has_email")
        
        # Extract phone numbers
        phones = re.findall(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', text)
        if phones:
            tags.add("#has_phone")
        
        return tags
    
    def _find_related_items(self, item: InformationItem, max_items: int = 5) -> List[str]:
        """Find items related to the given item."""
        related = []
        
        # Find items in same category
        category_items = self.category_index.get(item.category, [])
        for item_id in category_items[-10:]:  # Check recent items
            if item_id != item.id:
                related.append(item_id)
        
        # Find items with overlapping tags
        for tag in item.tags:
            tag_items = self.tag_index.get(tag, [])
            for item_id in tag_items[-5:]:
                if item_id != item.id and item_id not in related:
                    related.append(item_id)
        
        return related[:max_items]
    
    def retrieve(self,
                 query: Optional[str] = None,
                 category: Optional[str] = None,
                 source: Optional[str] = None,
                 tags: Optional[List[str]] = None,
                 time_range: Optional[Tuple[datetime, datetime]] = None,
                 limit: int = 10) -> List[InformationItem]:
        """
        Retrieve organized information based on criteria.
        
        Args:
            query: Text query to search for
            category: Filter by category
            source: Filter by source
            tags: Filter by tags
            time_range: Filter by time range
            limit: Maximum number of items to return
            
        Returns:
            List of matching InformationItems
        """
        candidates = []
        
        # Start with all items or filtered sets
        if category:
            candidates = [self.items[id] for id in self.category_index.get(category, [])]
        elif source:
            candidates = [self.items[id] for id in self.source_index.get(source, [])]
        elif tags:
            tag_items = set()
            for tag in tags:
                tag_items.update(self.tag_index.get(tag, []))
            candidates = [self.items[id] for id in tag_items]
        else:
            candidates = list(self.items.values())
        
        # Filter by time range
        if time_range:
            start, end = time_range
            candidates = [
                item for item in candidates
                if start <= item.timestamp <= end
            ]
        
        # Search by query
        if query:
            query_lower = query.lower()
            scored_items = []
            for item in candidates:
                content_str = str(item.content).lower()
                score = 0
                
                # Exact match
                if query_lower in content_str:
                    score += 1.0
                
                # Word matches
                query_words = query_lower.split()
                for word in query_words:
                    if word in content_str:
                        score += 0.5
                
                if score > 0:
                    scored_items.append((score, item))
            
            # Sort by score
            scored_items.sort(key=lambda x: x[0], reverse=True)
            candidates = [item for _, item in scored_items]
        
        # Update access counts
        for item in candidates[:limit]:
            item.access_count += 1
            item.last_accessed = datetime.now()
        
        return candidates[:limit]
    
    def get_summary(self, time_range: Optional[timedelta] = None) -> Dict[str, Any]:
        """
        Get a summary of organized information.
        
        Args:
            time_range: Optional time range to summarize
            
        Returns:
            Summary dictionary
        """
        summary = {
            "total_items": self.stats["total_items"],
            "categories": {},
            "sources": dict(self.stats["items_by_source"]),
            "top_tags": [],
            "recent_activity": []
        }
        
        # Category breakdown
        for cat_name, count in self.stats["items_by_category"].items():
            summary["categories"][cat_name] = {
                "count": count,
                "percentage": (count / self.stats["total_items"]) * 100 if self.stats["total_items"] > 0 else 0
            }
        
        # Top tags
        tag_counts = Counter()
        for tag, item_ids in self.tag_index.items():
            tag_counts[tag] = len(item_ids)
        summary["top_tags"] = tag_counts.most_common(10)
        
        # Recent activity
        if time_range:
            cutoff = datetime.now() - time_range
            recent_items = [
                self.items[item_id]
                for timestamp, item_id in reversed(self.time_index)
                if timestamp >= cutoff
            ]
            summary["recent_activity"] = [
                {
                    "category": item.category,
                    "source": item.source,
                    "timestamp": item.timestamp.isoformat(),
                    "preview": str(item.content)[:100]
                }
                for item in recent_items[:10]
            ]
        
        return summary
    
    def suggest_categories(self, text: str) -> List[Tuple[str, float]]:
        """
        Suggest categories for a piece of text.
        
        Returns:
            List of (category, confidence) tuples
        """
        suggestions = []
        
        all_categories = {**self.categories, **self.custom_categories}
        for cat_name, category in all_categories.items():
            score = category.matches(text)
            if score > 0.1:
                suggestions.append((cat_name, score))
        
        suggestions.sort(key=lambda x: x[1], reverse=True)
        return suggestions[:5]
    
    def save_state(self):
        """Save the current state to disk."""
        try:
            # Save items
            items_file = self.save_path / "items.jsonl"
            with open(items_file, "w") as f:
                for item in self.items.values():
                    f.write(json.dumps(item.to_dict()) + "\n")
            
            # Save statistics
            stats_file = self.save_path / "stats.json"
            with open(stats_file, "w") as f:
                json.dump({
                    "total_items": self.stats["total_items"],
                    "items_by_category": dict(self.stats["items_by_category"]),
                    "items_by_source": dict(self.stats["items_by_source"])
                }, f, indent=2)
            
            log.debug(f"Saved state: {self.stats['total_items']} items")
        
        except Exception as e:
            log.error(f"Failed to save state: {e}")
    
    def load_state(self):
        """Load previous state from disk."""
        items_file = self.save_path / "items.jsonl"
        if items_file.exists():
            with open(items_file, "r") as f:
                for line in f:
                    data = json.loads(line)
                    item = InformationItem(
                        content=data["content"],
                        source=data["source"],
                        category=data["category"],
                        timestamp=datetime.fromisoformat(data["timestamp"]),
                        metadata=data.get("metadata", {})
                    )
                    item.id = data["id"]
                    item.tags = set(data.get("tags", []))
                    item.relevance_score = data.get("relevance_score", 0)
                    item.access_count = data.get("access_count", 0)
                    
                    self.items[item.id] = item
                    self.category_index[item.category].append(item.id)
                    self.source_index[item.source].append(item.id)
                    for tag in item.tags:
                        self.tag_index[tag].append(item.id)
            
            log.info(f"Loaded {len(self.items)} items from state")