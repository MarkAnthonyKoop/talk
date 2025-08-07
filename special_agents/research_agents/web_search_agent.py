#!/usr/bin/env python3
# special_agents/web_search_agent.py

"""
WebSearchAgent - Specialized agent for performing web searches to gather research information.

This agent performs web searches using various search engines and APIs to gather
relevant information for coding tasks, documentation, libraries, and best practices.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Dict, List, Optional, Tuple, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from agent.agent import Agent
from urllib.parse import quote_plus
import requests
from bs4 import BeautifulSoup

from agent.agent import Agent
from agent.messages import Message, Role

log = logging.getLogger(__name__)

class WebSearchAgent(Agent):
    """
    Specialized agent for web search and research.
    
    This agent can search the web for coding-related information,
    documentation, libraries, best practices, and examples.
    """
    
    def __init__(self, max_results: int = 5, **kwargs):
        """Initialize with specialized system prompt for web search."""
        roles = kwargs.pop("roles", [])
        self.max_results = max_results
        
        # Add web search-specific system prompts
        search_system_prompts = [
            "You are a research assistant specialized in finding coding and technical information.",
            "You help developers by searching for relevant documentation, libraries, examples, and best practices.",
            "Always summarize your findings clearly and provide actionable insights.",
            "Focus on official documentation, reputable sources, and current information."
        ]
        
        # Combine with any existing roles
        roles = search_system_prompts + roles
        
        # Initialize the base agent
        super().__init__(roles=roles, **kwargs)
    
    def run(self, input_text: str) -> str:
        """
        Process search request and return relevant information.
        
        Args:
            input_text: Search query or research request
            
        Returns:
            Formatted search results and analysis
        """
        try:
            # Extract search intent and queries
            search_queries = self._extract_search_queries(input_text)
            
            # Perform searches
            all_results = []
            for query in search_queries:
                results = self._perform_search(query)
                all_results.extend(results)
            
            # Remove duplicates and limit results
            unique_results = self._deduplicate_results(all_results)
            limited_results = unique_results[:self.max_results]
            
            # Format results for consumption
            formatted_results = self._format_search_results(limited_results)
            
            # Generate summary and recommendations
            summary = self._generate_summary(input_text, formatted_results)
            
            return f"{summary}\n\n{formatted_results}"
            
        except Exception as e:
            log.error(f"Web search failed: {e}")
            return f"WEB_SEARCH_ERROR: {str(e)}\n\nFallback: Consider checking official documentation manually."
    
    def _extract_search_queries(self, input_text: str) -> List[str]:
        """
        Extract search queries from the input text.
        
        Args:
            input_text: The research request
            
        Returns:
            List of search queries to execute
        """
        # Create a prompt for the LLM to extract search queries
        extraction_prompt = f"""
Based on this research request, generate 2-3 specific search queries that would help find relevant information:

Request: {input_text}

Return only the search queries, one per line, without any additional text.
Focus on finding:
1. Official documentation
2. Code examples
3. Best practices
4. Libraries/frameworks

Search queries:
"""
        
        try:
            # Use the base agent to generate search queries
            response = super().run(extraction_prompt)
            
            # Parse the response to extract queries
            queries = []
            for line in response.strip().split('\n'):
                line = line.strip()
                if line and not line.startswith('#') and len(line) > 3:
                    # Clean up the query
                    query = re.sub(r'^[0-9]+\.\s*', '', line)  # Remove numbering
                    query = re.sub(r'^[-*]\s*', '', query)     # Remove bullet points
                    if query:
                        queries.append(query)
            
            # Fallback if no queries extracted
            if not queries:
                # Create default queries based on keywords
                keywords = self._extract_keywords(input_text)
                for keyword in keywords[:3]:
                    queries.append(f"{keyword} documentation examples")
            
            return queries[:3]  # Limit to 3 queries
            
        except Exception as e:
            log.warning(f"Query extraction failed: {e}")
            # Fallback to simple keyword extraction
            keywords = self._extract_keywords(input_text)
            return [f"{kw} documentation" for kw in keywords[:2]]
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract programming keywords from text."""
        # Common programming terms to look for
        prog_terms = ['python', 'javascript', 'java', 'react', 'node', 'api', 'database', 
                     'sql', 'html', 'css', 'framework', 'library', 'flask', 'django',
                     'express', 'vue', 'angular', 'docker', 'kubernetes', 'git']
        
        words = re.findall(r'\b\w+\b', text.lower())
        keywords = []
        
        for word in words:
            if word in prog_terms or len(word) > 6:  # Long words are often important
                if word not in keywords:
                    keywords.append(word)
        
        return keywords[:5]
    
    def _perform_search(self, query: str) -> List[Dict]:
        """
        Perform actual web search using DuckDuckGo.
        
        Args:
            query: Search query string
            
        Returns:
            List of search results
        """
        try:
            # Try DuckDuckGo search first (no API key required)
            results = self._duckduckgo_search(query)
            if results:
                log.info(f"DuckDuckGo search successful for: {query}")
                return results
        except Exception as e:
            log.warning(f"DuckDuckGo search failed for '{query}': {e}")
        
        try:
            # Fallback to direct HTTP search
            results = self._http_search(query)
            if results:
                log.info(f"HTTP search successful for: {query}")
                return results
        except Exception as e:
            log.warning(f"HTTP search failed for '{query}': {e}")
        
        # Final fallback to simulated results
        log.info(f"Falling back to simulated search for: {query}")
        return self._simulated_search(query)
    
    def _duckduckgo_search(self, query: str) -> List[Dict]:
        """Perform search using DuckDuckGo API."""
        try:
            # Try the new ddgs package first
            try:
                from ddgs import DDGS
            except ImportError:
                # Fall back to old package name if new one not installed
                from duckduckgo_search import DDGS
            
            results = []
            with DDGS() as ddgs:
                # Get up to 5 search results
                search_results = ddgs.text(query, max_results=5)
                
                for result in search_results:
                    results.append({
                        "title": result.get("title", "No title"),
                        "url": result.get("href", ""),
                        "snippet": result.get("body", "No description available"),
                        "source": "duckduckgo"
                    })
            
            return results
            
        except ImportError:
            log.warning("ddgs/duckduckgo-search not available, trying alternative methods")
            return []
        except Exception as e:
            log.error(f"DuckDuckGo search error: {e}")
            return []
    
    def _http_search(self, query: str) -> List[Dict]:
        """Perform search using direct HTTP requests to search engines."""
        results = []
        
        try:
            # Search Stack Overflow
            so_results = self._search_stackoverflow(query)
            results.extend(so_results)
            
            # Search GitHub
            gh_results = self._search_github(query)
            results.extend(gh_results)
            
            # Add a small delay to be respectful
            time.sleep(0.5)
            
        except Exception as e:
            log.error(f"HTTP search error: {e}")
        
        return results[:5]  # Limit to 5 results
    
    def _search_stackoverflow(self, query: str) -> List[Dict]:
        """Search Stack Overflow API."""
        try:
            # Use Stack Overflow API (no auth required for basic search)
            url = "https://api.stackexchange.com/2.3/search/advanced"
            params = {
                'order': 'desc',
                'sort': 'relevance',
                'q': query,
                'site': 'stackoverflow',
                'pagesize': 3
            }
            
            headers = {
                'User-Agent': 'TalkFramework/1.0 (Educational Research Tool)'
            }
            
            response = requests.get(url, params=params, headers=headers, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            for item in data.get('items', []):
                results.append({
                    "title": item.get('title', 'Stack Overflow Result'),
                    "url": item.get('link', ''),
                    "snippet": BeautifulSoup(item.get('body', ''), 'html.parser').get_text()[:200] + "...",
                    "source": "stackoverflow",
                    "score": item.get('score', 0),
                    "tags": item.get('tags', [])
                })
            
            return results
            
        except Exception as e:
            log.warning(f"Stack Overflow search failed: {e}")
            return []
    
    def _search_github(self, query: str) -> List[Dict]:
        """Search GitHub repositories."""
        try:
            # Use GitHub API (no auth required for basic search)
            url = "https://api.github.com/search/repositories"
            params = {
                'q': query + ' language:python',  # Focus on Python repos
                'sort': 'stars',
                'order': 'desc',
                'per_page': 2
            }
            
            headers = {
                'User-Agent': 'TalkFramework/1.0 (Educational Research Tool)',
                'Accept': 'application/vnd.github.v3+json'
            }
            
            response = requests.get(url, params=params, headers=headers, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            for item in data.get('items', []):
                results.append({
                    "title": f"GitHub: {item.get('name', 'Repository')}",
                    "url": item.get('html_url', ''),
                    "snippet": item.get('description', 'No description available'),
                    "source": "github",
                    "stars": item.get('stargazers_count', 0),
                    "language": item.get('language', 'Unknown')
                })
            
            return results
            
        except Exception as e:
            log.warning(f"GitHub search failed: {e}")
            return []
    
    def _simulated_search(self, query: str) -> List[Dict]:
        """Fallback simulated search results."""
        return [
            {
                "title": f"Documentation for {query}",
                "url": f"https://docs.example.com/{quote_plus(query)}",
                "snippet": f"Official documentation and examples for {query}. (Simulated result - live search unavailable)",
                "source": "simulated"
            },
            {
                "title": f"{query} Tutorial",
                "url": f"https://tutorial.example.com/{quote_plus(query)}",
                "snippet": f"Tutorial and examples for {query}. (Simulated result - live search unavailable)",
                "source": "simulated"
            }
        ]
    
    def _deduplicate_results(self, results: List[Dict]) -> List[Dict]:
        """Remove duplicate search results based on URL."""
        seen_urls = set()
        unique_results = []
        
        for result in results:
            url = result.get('url', '')
            if url not in seen_urls:
                seen_urls.add(url)
                unique_results.append(result)
        
        return unique_results
    
    def _format_search_results(self, results: List[Dict]) -> str:
        """Format search results for display."""
        if not results:
            return "No search results found."
        
        formatted = "SEARCH_RESULTS:\n\n"
        
        for i, result in enumerate(results, 1):
            title = result.get('title', 'Untitled')
            url = result.get('url', '')
            snippet = result.get('snippet', 'No description available')
            source = result.get('source', 'unknown')
            
            formatted += f"{i}. {title}\n"
            formatted += f"   URL: {url}\n"
            formatted += f"   Source: {source}\n"
            formatted += f"   Summary: {snippet}\n\n"
        
        return formatted.strip()
    
    def _generate_summary(self, original_request: str, search_results: str) -> str:
        """Generate a summary and recommendations based on search results."""
        summary_prompt = f"""
Based on the following search results for the research request, provide a concise summary and actionable recommendations:

Original Request: {original_request}

Search Results:
{search_results}

Please provide:
1. A brief summary of what was found
2. Key recommendations for the developer
3. Next steps or specific resources to explore

Keep it practical and actionable.

RESEARCH_SUMMARY:
"""
        
        try:
            # Use the base agent to generate summary
            summary = super().run(summary_prompt)
            return summary.strip()
        except Exception as e:
            log.warning(f"Summary generation failed: {e}")
            return "RESEARCH_SUMMARY: Search completed. Review the results below for relevant information."


class WebSearchAgentIntegration:
    """
    Integration helper for WebSearchAgent with Talk framework.
    """
    
    @staticmethod
    def should_use_web_search(task_description: str, llm_agent: Optional['Agent'] = None) -> bool:
        """
        Intelligently determine if a task would benefit from web search using LLM analysis.
        
        Args:
            task_description: The coding task description
            llm_agent: Optional LLM agent for intelligent analysis
            
        Returns:
            True if web search would be helpful
        """
        # If no LLM agent provided, fall back to simple heuristics
        if llm_agent is None:
            return WebSearchAgentIntegration._simple_heuristic_check(task_description)
        
        # Use LLM to intelligently analyze the task
        analysis_prompt = f"""
Analyze this coding task and determine if web search/research would be helpful:

Task: "{task_description}"

IMPORTANT: Only suggest research for tasks that truly need external information.

Tasks that DO NOT need research (answer NO):
- Fixing typos, syntax errors, or simple bugs
- Renaming variables, functions, or files  
- Adding comments or documentation
- Simple refactoring or code cleanup
- Basic debugging of existing code
- Minor modifications to existing functionality

Tasks that DO need research (answer YES):
- Learning new frameworks, libraries, or technologies
- Implementing complex features with multiple systems
- Finding best practices for unfamiliar domains
- Integration between different technologies
- Modern approaches to well-known problems

Be conservative - only suggest research when the task genuinely requires external knowledge that a skilled developer wouldn't already know.

Respond with exactly "YES" or "NO":"""

        try:
            response = llm_agent.run(analysis_prompt)
            decision = response.strip().upper()
            
            # Extract YES/NO from response
            if "YES" in decision:
                return True
            elif "NO" in decision:
                return False
            else:
                # If unclear response, fall back to heuristics
                log.warning(f"Unclear LLM response for search decision: {response}")
                return WebSearchAgentIntegration._simple_heuristic_check(task_description)
                
        except Exception as e:
            log.warning(f"LLM-based search decision failed: {e}")
            # Fall back to simple heuristics
            return WebSearchAgentIntegration._simple_heuristic_check(task_description)
    
    @staticmethod
    def _simple_heuristic_check(task_description: str) -> bool:
        """
        Simple fallback heuristics for search decision.
        
        Args:
            task_description: The coding task description
            
        Returns:
            True if likely to benefit from search
        """
        task_lower = task_description.lower()
        
        # Definite YES indicators
        research_words = ['research', 'find', 'documentation', 'tutorial', 'example', 
                         'best practice', 'how to', 'unfamiliar', 'unknown']
        if any(word in task_lower for word in research_words):
            return True
        
        # Definite NO indicators (simple tasks)
        simple_tasks = ['fix typo', 'fix bug', 'debug', 'refactor', 'rename', 
                       'update comment', 'add comment', 'remove comment']
        if any(simple in task_lower for simple in simple_tasks):
            return False
        
        # Complex action words that suggest research might help
        complex_actions = ['build', 'create', 'implement', 'integrate', 'setup', 
                          'configure', 'develop', 'design']
        action_found = any(action in task_lower for action in complex_actions)
        
        # Technology words that often need research
        tech_words = ['api', 'authentication', 'database', 'framework', 'library']
        tech_found = any(tech in task_lower for tech in tech_words)
        
        # If both complex action AND technology mentioned, likely needs research
        return action_found and tech_found
    
    @staticmethod
    def create_search_query(task_description: str) -> str:
        """
        Create a search query from a task description.
        
        Args:
            task_description: The coding task description
            
        Returns:
            Formatted search query
        """
        return f"Research and find information to help with this coding task: {task_description}"


# For backward compatibility and easy imports
__all__ = ['WebSearchAgent', 'WebSearchAgentIntegration']