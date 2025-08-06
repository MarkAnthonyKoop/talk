#!/usr/bin/env python3
"""
Direct test to start Serena MCP server and keep dashboard active.

This bypasses the health check and starts the server directly.
"""

import sys
import time
import signal
from pathlib import Path

# Add the project root to path so we can import our modules
sys.path.append(str(Path(__file__).parent))

from special_agents.reminiscing.serena_wrapper import SerenaWrapper

def start_dashboard_server():
    """Start Serena server directly and keep it running."""
    print("Starting Serena MCP Server with Dashboard (Direct Method)")
    print("=" * 60)
    
    try:
        # Create wrapper
        serena = SerenaWrapper()
        
        # Test that Serena is available
        test_result = serena.test_installation()
        print("Serena Status:", "✅ Available" if test_result["success"] else "❌ Not Available")
        
        if not test_result["success"]:
            print("Error:", test_result["message"])
            return
        
        project_path = str(Path.cwd())
        port = 9121
        
        print(f"\n🚀 Starting MCP Server:")
        print(f"   Project: {project_path}")
        print(f"   Port: {port}")
        print(f"   Context: ide-assistant")
        print(f"   Dashboard: Should auto-open at http://127.0.0.1:24282/dashboard/index.html")
        
        # Start the server
        server_process = serena.start_mcp_server(
            project_path=project_path,
            port=port,
            context="ide-assistant",
            mode=["interactive", "editing"]
        )
        
        print(f"\n✅ Server started! Process ID: {server_process.pid}")
        print("\n📊 Dashboard should now be accessible.")
        print("   The dashboard will show real-time activity as the server runs.")
        
        # Set up signal handler for clean shutdown
        def signal_handler(sig, frame):
            print(f"\n\n🛑 Received signal {sig}, shutting down server...")
            server_process.terminate()
            try:
                server_process.wait(timeout=5)
                print("✅ Server shut down cleanly")
            except:
                print("⚠️  Server killed forcefully")
                server_process.kill()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        print("\n" + "=" * 60)
        print("🌐 Dashboard Information:")
        print("   URL: http://127.0.0.1:24282/dashboard/index.html")
        print("   - View real-time logs")
        print("   - Monitor MCP tool usage")  
        print("   - See server configuration")
        print("   - Use shutdown button for clean termination")
        
        print("\n⏸️  Server is running... Press Ctrl+C to stop")
        print("   (or use the shutdown button in the dashboard)")
        
        # Keep the server running and show status
        while True:
            if server_process.poll() is not None:
                print(f"\n⚠️  Server process ended with code: {server_process.poll()}")
                break
            
            time.sleep(5)
            print(".", end="", flush=True)  # Show we're still alive
            
    except KeyboardInterrupt:
        print(f"\n\n🛑 Interrupted by user")
        if 'server_process' in locals():
            server_process.terminate()
            try:
                server_process.wait(timeout=5)
            except:
                server_process.kill()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    start_dashboard_server()