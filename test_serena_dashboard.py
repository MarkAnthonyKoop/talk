#!/usr/bin/env python3
"""
Test script to start Serena MCP server and keep dashboard active for exploration.

This will start the server, open the dashboard, and wait for user input
before shutting down, allowing you to explore Serena's web interface.
"""

import sys
import time
from pathlib import Path

# Add the project root to path so we can import our modules
sys.path.append(str(Path(__file__).parent))

from special_agents.reminiscing.serena_integration_agent import SerenaIntegrationAgent

def test_dashboard_exploration():
    """Start Serena server and keep dashboard active for exploration."""
    print("Starting Serena MCP Server with Dashboard")
    print("=" * 50)
    
    try:
        # Create the integration agent
        agent = SerenaIntegrationAgent(name="DashboardTestAgent")
        
        print("✅ SerenaIntegrationAgent created successfully")
        
        # Start the server for the current project
        project_path = str(Path.cwd())
        print(f"\n🚀 Starting Serena MCP server for project: {project_path}")
        
        response = agent.run(f"start serena {project_path}")
        print(response)
        
        # Check if server started successfully
        if "SERENA_STARTED" in response:
            print("\n" + "=" * 50)
            print("🌐 Dashboard should now be available at:")
            print("   http://127.0.0.1:24282/dashboard/index.html")
            print("\n📊 The dashboard shows:")
            print("   - Real-time MCP server logs")
            print("   - Tool usage statistics")
            print("   - Server status and controls")
            print("   - Shutdown button for clean server termination")
            
            print("\n🔍 You can now:")
            print("   1. Explore the dashboard interface")
            print("   2. See how Serena monitors MCP activity")
            print("   3. Try some semantic search commands")
            
            # Demonstrate some MCP activity to show in dashboard
            print("\n⚡ Generating some activity for the dashboard...")
            
            # Try a few operations to generate log activity
            demo_commands = [
                "What is the current server status?",
                "find function Agent in the codebase",
                "analyze symbol SerenaIntegrationAgent",
                "get codebase overview"
            ]
            
            for i, cmd in enumerate(demo_commands, 1):
                print(f"\n   Demo {i}: {cmd}")
                response = agent.run(cmd)
                # Just show first line of response to keep output clean
                first_line = response.split('\n')[0] if response else "No response"
                print(f"   Response: {first_line}")
                time.sleep(1)  # Brief pause between commands
            
            print("\n" + "=" * 50)
            print("🎯 Dashboard Activity Generated!")
            print("   Check the dashboard to see the logs and activity")
            print("   The server will stay running until you press Enter...")
            
            # Keep server running until user input
            try:
                input("\n⏸️  Press Enter to stop the server and close dashboard...")
            except KeyboardInterrupt:
                print("\n\n🛑 Interrupted by user")
            
            # Stop the server
            print("\n🛑 Stopping Serena MCP server...")
            stop_response = agent.run("stop serena")
            print(stop_response)
            
        else:
            print("❌ Server failed to start properly")
            print("Response:", response)
            
        print("\n✅ Test completed!")
        
    except Exception as e:
        print(f"❌ Error in dashboard test: {e}")
        import traceback
        traceback.print_exc()

def show_dashboard_info():
    """Show information about what to look for in the dashboard."""
    print("\n📋 Dashboard Features to Explore:")
    print("=" * 50)
    print("""
🏠 Main Dashboard:
   - Server status and uptime
   - Current project information  
   - Active tools and configurations
   
📝 Logs Section:
   - Real-time MCP protocol messages
   - Tool execution logs
   - Error messages and debugging info
   
📊 Statistics (if enabled):
   - Tool usage frequency
   - Performance metrics
   - Request/response timing
   
🔧 Controls:
   - Server shutdown button
   - Configuration display
   - Session management
   
💡 What This Shows:
   - How Serena provides semantic code analysis
   - MCP protocol communication in action  
   - Tool orchestration for focused context retrieval
   - The infrastructure solving the "30% potential" problem
""")

if __name__ == "__main__":
    show_dashboard_info()
    test_dashboard_exploration()