#!/usr/bin/env python3
"""
VeriGuard AI - LLM Hallucination Detection & Verification Engine

Usage:
    python main.py                    # Interactive mode
    python main.py --api              # Start REST API server
    python main.py --verify "text"    # Single verification
    python main.py --rebuild          # Rebuild knowledge base index

Author: Keerthi Adapa
License: MIT
"""

import argparse
import sys

from config.settings import settings
from core.pipeline import get_pipeline
from core.models import VerificationRequest, VerificationStatus, SeverityLevel
from utils.logger import get_logger

logger = get_logger("main")


def print_banner():
    """Display simple banner."""
    print("\n" + "="*60)
    print("  VeriGuard AI - Hallucination Detection Engine v2.0.0")
    print("="*60 + "\n")


def display_results(response):
    """Display verification results."""
    print("\n" + "-"*60)
    print("VERIFICATION RESULTS")
    print("-"*60)
    print(f"Request ID: {response.request_id}")
    print(f"Total Claims: {response.total_claims}")
    print(f"Hallucination Rate: {response.hallucination_rate:.1%}")
    print(f"Risk Score: {response.overall_risk_score:.2f}")
    print(f"Processing Time: {response.processing_time_ms:.0f}ms")
    print("-"*60)
    
    if response.analyses:
        print("\nCLAIMS:")
        for i, analysis in enumerate(response.analyses, 1):
            status = analysis.verification.status.value
            claim_text = analysis.claim.text
            
            # Simple status indicators
            if analysis.verification.status == VerificationStatus.SUPPORTED:
                indicator = "[OK]"
            elif analysis.verification.status == VerificationStatus.CONTRADICTED:
                indicator = "[FALSE]"
            elif analysis.verification.status == VerificationStatus.PARTIALLY_SUPPORTED:
                indicator = "[PARTIAL]"
            else:
                indicator = "[?]"
            
            print(f"\n  {i}. {indicator} {claim_text}")
            print(f"     Status: {status}")
            print(f"     Confidence: {analysis.verification.confidence:.0%}")
            print(f"     Risk: {analysis.risk.severity.value} ({analysis.risk.risk_score:.2f})")
            
            if analysis.verification.explanation:
                print(f"     Reason: {analysis.verification.explanation}")
    
    print("\n" + "="*60 + "\n")


def interactive_mode():
    """Run interactive CLI mode."""
    print_banner()
    
    print("Loading system...")
    pipeline = get_pipeline(auto_init_kb=True)
    print("Ready!\n")
    
    print("Enter text to verify (type 'quit' to exit, 'help' for commands):\n")
    
    while True:
        try:
            text = input(">>> ").strip()
            
            if not text:
                continue
            
            if text.lower() == 'quit':
                print("Goodbye!")
                break
            
            if text.lower() == 'help':
                print("""
Commands:
  quit   - Exit the program
  help   - Show this message
  stats  - Show knowledge base stats

Usage:
  Just type any text and press Enter to check for hallucinations.
""")
                continue
            
            if text.lower() == 'stats':
                from core.index import get_knowledge_index
                index = get_knowledge_index()
                stats = index.get_stats()
                print(f"\nKnowledge Base: {stats['total_documents']} documents")
                for cat, count in stats['categories'].items():
                    print(f"  - {cat}: {count}")
                print()
                continue
            
            # Process verification
            print("\nAnalyzing...")
            
            request = VerificationRequest(
                text=text,
                extraction_mode="standard",
                verification_depth="standard",
                include_evidence=False,
                include_explanations=True
            )
            
            response = pipeline.verify(request)
            display_results(response)
            
        except KeyboardInterrupt:
            print("\nType 'quit' to exit.")
        except Exception as e:
            print(f"Error: {e}")
            logger.exception("Error in interactive mode")


def verify_single(text: str, output_json: bool = False):
    """Verify a single text from command line."""
    pipeline = get_pipeline(auto_init_kb=True)
    
    request = VerificationRequest(
        text=text,
        extraction_mode="standard",
        verification_depth="standard"
    )
    
    response = pipeline.verify(request)
    
    if output_json:
        print(response.model_dump_json(indent=2))
    else:
        display_results(response)


def start_api_server():
    """Start the FastAPI server."""
    import uvicorn
    from api.app import app
    
    print(f"\nStarting VeriGuard AI API server...")
    print(f"  Host: {settings.api.host}")
    print(f"  Port: {settings.api.port}")
    print(f"  Docs: http://localhost:{settings.api.port}/docs\n")
    
    uvicorn.run(
        app,
        host=settings.api.host,
        port=settings.api.port,
        log_level="info"
    )


def rebuild_index():
    """Rebuild the knowledge base index."""
    from core.index import initialize_knowledge_base
    
    print("Rebuilding knowledge base index...")
    doc_count = initialize_knowledge_base(force_rebuild=True)
    print(f"Done! Indexed {doc_count} documents")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="VeriGuard AI - LLM Hallucination Detection Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                           # Interactive mode
  python main.py --api                     # Start REST API
  python main.py --verify "some text"      # Verify text
  python main.py --verify "text" --json    # JSON output
  python main.py --rebuild                 # Rebuild index
        """
    )
    
    parser.add_argument(
        "--api", 
        action="store_true",
        help="Start the REST API server"
    )
    parser.add_argument(
        "--verify", 
        type=str,
        metavar="TEXT",
        help="Verify a single text"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON (use with --verify)"
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild the knowledge base index"
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"VeriGuard AI {settings.version}"
    )
    
    args = parser.parse_args()
    
    try:
        if args.api:
            start_api_server()
        elif args.verify:
            verify_single(args.verify, args.json)
        elif args.rebuild:
            rebuild_index()
        else:
            interactive_mode()
    except KeyboardInterrupt:
        print("\nTerminated.")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        logger.exception("Fatal error")
        sys.exit(1)


if __name__ == "__main__":
    main()