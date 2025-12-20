#!/usr/bin/env python3
"""
Script to delete all nodes and relationships for a specific show from Neo4j database.

This script connects to a Neo4j database and removes all nodes and relationships
where the 'show' property equals the specified show name. It uses DETACH DELETE 
to ensure all connected relationships are properly removed.

The script loads Neo4j connection parameters from keys/.env file.

Usage:
    python delete_show_data.py SHOW_NAME [--uri URI] [--username USERNAME] [--password PASSWORD]
    
Examples:
    python delete_show_data.py bva
    python delete_show_data.py ecomm
    python delete_show_data.py --list-shows
    
Environment Variables (loaded from keys/.env):
    NEO4J_URI: Neo4j connection URI (default: bolt://localhost:7687)
    NEO4J_USERNAME: Neo4j username (default: neo4j)
    NEO4J_PASSWORD: Neo4j password (required)
    
Required Python packages:
    neo4j: pip install neo4j
    python-dotenv: pip install python-dotenv
"""

import argparse
import os
import sys
import logging
from typing import Dict, Any, Optional, Tuple, List, Set
from pathlib import Path

from neo4j import GraphDatabase, Driver
from neo4j.exceptions import Neo4jError, ServiceUnavailable, AuthError
from dotenv import load_dotenv

# Load environment variables from keys/.env
env_path = Path("keys/.env")
if env_path.exists():
    load_dotenv(env_path)
    logger_init = logging.getLogger(__name__)
    logger_init.info(f"Loaded environment variables from {env_path}")
else:
    # Try alternate path if script is run from different directory
    alt_env_path = Path(__file__).parent / "keys" / ".env"
    if alt_env_path.exists():
        load_dotenv(alt_env_path)
        logger_init = logging.getLogger(__name__)
        logger_init.info(f"Loaded environment variables from {alt_env_path}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ShowDataDeleter:
    """Handles deletion of show-specific data from Neo4j database."""
    
    def __init__(self, uri: str, username: str, password: str) -> None:
        """
        Initialize the show data deleter with database connection parameters.
        
        Args:
            uri: Neo4j database URI
            username: Database username
            password: Database password
        """
        self.uri = uri
        self.username = username
        self.password = password
        self.driver: Optional[Driver] = None
        
    def connect(self) -> None:
        """
        Establish connection to the Neo4j database.
        
        Raises:
            ServiceUnavailable: If the database is not reachable
            AuthError: If authentication fails
        """
        try:
            self.driver = GraphDatabase.driver(
                self.uri, 
                auth=(self.username, self.password)
            )
            # Verify connectivity
            self.driver.verify_connectivity()
            logger.info(f"Successfully connected to Neo4j at {self.uri}")
        except ServiceUnavailable as e:
            logger.error(f"Failed to connect to Neo4j at {self.uri}: {e}")
            raise
        except AuthError as e:
            logger.error(f"Authentication failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error connecting to database: {e}")
            raise
            
    def close(self) -> None:
        """Close the database connection."""
        if self.driver:
            self.driver.close()
            logger.info("Database connection closed")
            
    def list_available_shows(self) -> Set[str]:
        """
        List all unique show values in the database.
        
        Returns:
            Set of unique show names found in the database
        """
        if not self.driver:
            raise RuntimeError("Database connection not established")
            
        shows = set()
        
        with self.driver.session() as session:
            try:
                # Get unique show values from nodes
                node_result = session.run("""
                    MATCH (n)
                    WHERE n.show IS NOT NULL
                    RETURN DISTINCT n.show as show
                    ORDER BY show
                """)
                
                for record in node_result:
                    shows.add(record["show"])
                
                # Get unique show values from relationships
                rel_result = session.run("""
                    MATCH ()-[r]->()
                    WHERE r.show IS NOT NULL
                    RETURN DISTINCT r.show as show
                    ORDER BY show
                """)
                
                for record in rel_result:
                    shows.add(record["show"])
                    
            except Neo4jError as e:
                logger.error(f"Error listing shows: {e}")
                raise
                
        return shows
    
    def count_show_data(self, show_name: str) -> Dict[str, Any]:
        """
        Count all nodes and relationships for a specific show.
        
        Args:
            show_name: Name of the show to count data for
            
        Returns:
            Dictionary containing counts of nodes and relationships to be deleted
        """
        if not self.driver:
            raise RuntimeError("Database connection not established")
            
        counts = {
            "show_name": show_name,
            "nodes": {},
            "relationships": {},
            "total_nodes": 0,
            "total_relationships": 0
        }
        
        with self.driver.session() as session:
            try:
                # Count nodes by label
                node_result = session.run("""
                    MATCH (n)
                    WHERE n.show = $show_name
                    RETURN labels(n) as node_type, COUNT(n) as count
                """, show_name=show_name)
                
                for record in node_result:
                    node_type = record["node_type"][0] if record["node_type"] else "Unknown"
                    count = record["count"]
                    counts["nodes"][node_type] = count
                    counts["total_nodes"] += count
                
                # Count relationships by type with show property
                rel_result = session.run("""
                    MATCH ()-[r]->()
                    WHERE r.show = $show_name
                    RETURN type(r) as relationship_type, COUNT(r) as count
                """, show_name=show_name)
                
                for record in rel_result:
                    rel_type = record["relationship_type"]
                    count = record["count"]
                    counts["relationships"][rel_type] = count
                    counts["total_relationships"] += count
                    
                # Count relationships that will be deleted due to node deletion
                indirect_rel_result = session.run("""
                    MATCH (n)-[r]-()
                    WHERE n.show = $show_name
                    RETURN COUNT(DISTINCT r) as count
                """, show_name=show_name)
                
                indirect_count = indirect_rel_result.single()["count"]
                counts["indirect_relationships"] = indirect_count
                
            except Neo4jError as e:
                logger.error(f"Error counting {show_name} data: {e}")
                raise
                
        return counts
    
    def delete_show_data(self, show_name: str, dry_run: bool = False) -> Tuple[int, int]:
        """
        Delete all nodes and relationships for a specific show.
        
        Args:
            show_name: Name of the show to delete data for
            dry_run: If True, only simulate deletion without actually deleting
            
        Returns:
            Tuple of (nodes_deleted, relationships_deleted)
        """
        if not self.driver:
            raise RuntimeError("Database connection not established")
            
        if dry_run:
            logger.info(f"DRY RUN MODE - No actual deletion will occur for show '{show_name}'")
            counts = self.count_show_data(show_name)
            return counts["total_nodes"], counts["total_relationships"] + counts.get("indirect_relationships", 0)
            
        with self.driver.session() as session:
            try:
                # Delete all nodes with specified show and their relationships
                result = session.run("""
                    MATCH (n)
                    WHERE n.show = $show_name
                    DETACH DELETE n
                    RETURN COUNT(*) as nodes_deleted
                """, show_name=show_name)
                
                summary = result.consume()
                nodes_deleted = summary.counters.nodes_deleted
                relationships_deleted = summary.counters.relationships_deleted
                
                # Delete any remaining relationships with specified show property
                # (in case there are orphaned relationships)
                rel_result = session.run("""
                    MATCH ()-[r]->()
                    WHERE r.show = $show_name
                    DELETE r
                    RETURN COUNT(*) as relationships_deleted
                """, show_name=show_name)
                
                rel_summary = rel_result.consume()
                additional_rels_deleted = rel_summary.counters.relationships_deleted
                relationships_deleted += additional_rels_deleted
                
                logger.info(f"Deletion complete for show '{show_name}': {nodes_deleted} nodes and {relationships_deleted} relationships deleted")
                
                return nodes_deleted, relationships_deleted
                
            except Neo4jError as e:
                logger.error(f"Error deleting {show_name} data: {e}")
                raise
    
    def verify_deletion(self, show_name: str) -> bool:
        """
        Verify that all data for the specified show has been deleted.
        
        Args:
            show_name: Name of the show to verify deletion for
            
        Returns:
            True if all show data has been deleted, False otherwise
        """
        if not self.driver:
            raise RuntimeError("Database connection not established")
            
        with self.driver.session() as session:
            try:
                # Check for remaining nodes
                node_result = session.run("""
                    MATCH (n)
                    WHERE n.show = $show_name
                    RETURN COUNT(n) as count
                """, show_name=show_name)
                
                remaining_nodes = node_result.single()["count"]
                
                # Check for remaining relationships
                rel_result = session.run("""
                    MATCH ()-[r]->()
                    WHERE r.show = $show_name
                    RETURN COUNT(r) as count
                """, show_name=show_name)
                
                remaining_rels = rel_result.single()["count"]
                
                if remaining_nodes == 0 and remaining_rels == 0:
                    logger.info(f"Verification successful: All '{show_name}' data has been deleted")
                    return True
                else:
                    logger.warning(f"Verification failed for '{show_name}': {remaining_nodes} nodes and {remaining_rels} relationships remain")
                    return False
                    
            except Neo4jError as e:
                logger.error(f"Error verifying deletion for '{show_name}': {e}")
                raise
    
    def delete_all_shows(self, shows: List[str], dry_run: bool = False) -> Dict[str, Tuple[int, int]]:
        """
        Delete data for multiple shows.
        
        Args:
            shows: List of show names to delete
            dry_run: If True, only simulate deletion
            
        Returns:
            Dictionary mapping show names to (nodes_deleted, relationships_deleted) tuples
        """
        results = {}
        
        for show in shows:
            logger.info(f"Processing show: {show}")
            try:
                nodes_deleted, rels_deleted = self.delete_show_data(show, dry_run)
                results[show] = (nodes_deleted, rels_deleted)
                
                if not dry_run:
                    self.verify_deletion(show)
                    
            except Neo4jError as e:
                logger.error(f"Failed to delete data for show '{show}': {e}")
                results[show] = (0, 0)
                
        return results


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Delete all nodes and relationships for a specific show from Neo4j database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Delete BVA show data:
    python delete_show_data.py bva
    
  Delete ECOMM show data with dry run:
    python delete_show_data.py ecomm --dry-run
    
  Delete multiple shows:
    python delete_show_data.py bva ecomm other_show
    
  List all available shows:
    python delete_show_data.py --list-shows
    
  Delete all shows in database:
    python delete_show_data.py --all
    
Environment Variables (loaded from keys/.env):
  NEO4J_URI: Neo4j connection URI
  NEO4J_USERNAME: Neo4j username  
  NEO4J_PASSWORD: Neo4j password
        """
    )
    
    parser.add_argument(
        "show_names",
        nargs="*",
        help="Name(s) of the show(s) to delete (e.g., bva, ecomm). Can specify multiple shows."
    )
    
    parser.add_argument(
        "--uri",
        type=str,
        default=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        help="Neo4j connection URI (default: NEO4J_URI from keys/.env or bolt://localhost:7687)"
    )
    
    parser.add_argument(
        "--username",
        type=str,
        default=os.getenv("NEO4J_USERNAME", "neo4j"),
        help="Neo4j username (default: NEO4J_USERNAME from keys/.env or neo4j)"
    )
    
    parser.add_argument(
        "--password",
        type=str,
        default=os.getenv("NEO4J_PASSWORD"),
        help="Neo4j password (default: NEO4J_PASSWORD from keys/.env)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform a dry run without actually deleting data"
    )
    
    parser.add_argument(
        "--no-confirm",
        action="store_true",
        help="Skip confirmation prompt before deletion"
    )
    
    parser.add_argument(
        "--list-shows",
        action="store_true",
        help="List all available shows in the database"
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        help="Delete data for ALL shows in the database (use with caution!)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


def confirm_deletion(counts: Dict[str, Any]) -> bool:
    """
    Prompt user to confirm deletion operation.
    
    Args:
        counts: Dictionary containing counts of data to be deleted
        
    Returns:
        True if user confirms, False otherwise
    """
    show_name = counts["show_name"].upper()
    
    print("\n" + "="*60)
    print(f"{show_name} DATA DELETION SUMMARY")
    print("="*60)
    
    print("\nNodes to be deleted:")
    if counts["nodes"]:
        for node_type, count in counts["nodes"].items():
            print(f"  - {node_type}: {count:,}")
    else:
        print("  None")
    
    print(f"\nTotal nodes: {counts['total_nodes']:,}")
    
    print("\nRelationships to be deleted (with show property):")
    if counts["relationships"]:
        for rel_type, count in counts["relationships"].items():
            print(f"  - {rel_type}: {count:,}")
    else:
        print("  None")
    
    print(f"\nRelationships connected to nodes being deleted: {counts.get('indirect_relationships', 0):,}")
    print(f"Total relationships to be deleted: {counts['total_relationships'] + counts.get('indirect_relationships', 0):,}")
    
    print("\n" + "="*60)
    
    response = input(f"\nAre you sure you want to delete all '{counts['show_name']}' data? This action cannot be undone! (yes/no): ")
    return response.lower() in ["yes", "y"]


def confirm_multiple_deletion(all_counts: List[Dict[str, Any]]) -> bool:
    """
    Prompt user to confirm deletion of multiple shows.
    
    Args:
        all_counts: List of count dictionaries for each show
        
    Returns:
        True if user confirms, False otherwise
    """
    print("\n" + "="*60)
    print("MULTIPLE SHOWS DELETION SUMMARY")
    print("="*60)
    
    total_nodes = 0
    total_relationships = 0
    
    for counts in all_counts:
        show_name = counts["show_name"]
        nodes = counts["total_nodes"]
        rels = counts["total_relationships"] + counts.get("indirect_relationships", 0)
        
        total_nodes += nodes
        total_relationships += rels
        
        print(f"\n{show_name.upper()}:")
        print(f"  - Nodes: {nodes:,}")
        print(f"  - Relationships: {rels:,}")
    
    print("\n" + "-"*60)
    print(f"TOTAL: {total_nodes:,} nodes and {total_relationships:,} relationships")
    print("="*60)
    
    response = input(f"\nAre you sure you want to delete data for {len(all_counts)} show(s)? This action cannot be undone! (yes/no): ")
    return response.lower() in ["yes", "y"]


def main() -> int:
    """
    Main entry point for the script.
    
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    args = parse_arguments()
    
    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Log connection parameters (without password)
    logger.debug(f"Neo4j URI: {args.uri}")
    logger.debug(f"Neo4j Username: {args.username}")
    logger.debug(f"Password provided: {'Yes' if args.password else 'No'}")
    
    # Validate password
    if not args.password:
        logger.error("Neo4j password is required. Set NEO4J_PASSWORD in keys/.env file or use --password")
        logger.error("Expected .env file location: keys/.env")
        return 1
    
    # Initialize deleter
    deleter = ShowDataDeleter(args.uri, args.username, args.password)
    
    try:
        # Connect to database
        deleter.connect()
        
        # Handle --list-shows option
        if args.list_shows:
            logger.info("Listing all available shows in the database...")
            shows = deleter.list_available_shows()
            
            if shows:
                print("\nAvailable shows in the database:")
                for show in sorted(shows):
                    # Get count for each show
                    counts = deleter.count_show_data(show)
                    print(f"  - {show}: {counts['total_nodes']:,} nodes, "
                          f"{counts['total_relationships'] + counts.get('indirect_relationships', 0):,} relationships")
            else:
                print("\nNo shows found in the database.")
            return 0
        
        # Determine which shows to delete
        shows_to_delete = []
        
        if args.all:
            # Delete all shows
            logger.info("Fetching all shows from database...")
            shows_to_delete = list(deleter.list_available_shows())
            
            if not shows_to_delete:
                logger.info("No shows found in the database. Nothing to delete.")
                return 0
                
            logger.info(f"Found {len(shows_to_delete)} show(s): {', '.join(shows_to_delete)}")
            
        elif args.show_names:
            # Delete specified shows
            shows_to_delete = args.show_names
            
        else:
            # No shows specified
            parser.print_help()
            print("\nError: Please specify show name(s) to delete, use --all, or use --list-shows")
            return 1
        
        # Process each show
        if len(shows_to_delete) == 1:
            # Single show deletion
            show_name = shows_to_delete[0]
            
            # Count data to be deleted
            logger.info(f"Analyzing '{show_name}' data in database...")
            counts = deleter.count_show_data(show_name)
            
            if counts["total_nodes"] == 0 and counts["total_relationships"] == 0:
                logger.info(f"No '{show_name}' data found in the database. Nothing to delete.")
                return 0
            
            # Confirm deletion unless --no-confirm is set
            if not args.no_confirm and not args.dry_run:
                if not confirm_deletion(counts):
                    logger.info("Deletion cancelled by user")
                    return 0
            
            # Perform deletion
            logger.info(f"Starting deletion process for '{show_name}'...")
            nodes_deleted, relationships_deleted = deleter.delete_show_data(show_name, dry_run=args.dry_run)
            
            if args.dry_run:
                logger.info(f"DRY RUN: Would delete {nodes_deleted:,} nodes and {relationships_deleted:,} relationships for '{show_name}'")
            else:
                logger.info(f"Successfully deleted {nodes_deleted:,} nodes and {relationships_deleted:,} relationships for '{show_name}'")
                
                # Verify deletion
                if deleter.verify_deletion(show_name):
                    logger.info(f"Deletion verification passed for '{show_name}'")
                else:
                    logger.warning(f"Some '{show_name}' data may still remain in the database")
                    return 1
        else:
            # Multiple shows deletion
            logger.info(f"Analyzing data for {len(shows_to_delete)} show(s)...")
            
            all_counts = []
            for show in shows_to_delete:
                counts = deleter.count_show_data(show)
                if counts["total_nodes"] > 0 or counts["total_relationships"] > 0:
                    all_counts.append(counts)
                else:
                    logger.info(f"No data found for show '{show}'")
            
            if not all_counts:
                logger.info("No data found for any of the specified shows. Nothing to delete.")
                return 0
            
            # Confirm deletion unless --no-confirm is set
            if not args.no_confirm and not args.dry_run:
                if not confirm_multiple_deletion(all_counts):
                    logger.info("Deletion cancelled by user")
                    return 0
            
            # Perform deletion for all shows
            logger.info(f"Starting deletion process for {len(all_counts)} show(s)...")
            results = deleter.delete_all_shows([c["show_name"] for c in all_counts], dry_run=args.dry_run)
            
            # Report results
            print("\n" + "="*60)
            print("DELETION RESULTS")
            print("="*60)
            
            total_nodes = 0
            total_rels = 0
            
            for show, (nodes, rels) in results.items():
                if args.dry_run:
                    print(f"{show}: Would delete {nodes:,} nodes and {rels:,} relationships")
                else:
                    print(f"{show}: Deleted {nodes:,} nodes and {rels:,} relationships")
                total_nodes += nodes
                total_rels += rels
            
            print("-"*60)
            if args.dry_run:
                print(f"TOTAL: Would delete {total_nodes:,} nodes and {total_rels:,} relationships")
            else:
                print(f"TOTAL: Deleted {total_nodes:,} nodes and {total_rels:,} relationships")
        
        return 0
        
    except (ServiceUnavailable, AuthError) as e:
        logger.error(f"Connection error: {e}")
        return 1
    except Neo4jError as e:
        logger.error(f"Database error: {e}")
        return 1
    except KeyboardInterrupt:
        logger.info("\nOperation cancelled by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1
    finally:
        deleter.close()


if __name__ == "__main__":
    sys.exit(main())