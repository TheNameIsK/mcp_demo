#!/usr/bin/env python3
"""
Unified MCP Server for Firebase and Supabase
Provides analytics and query capabilities for both Firebase Firestore and Supabase PostgreSQL
"""

import os
import re
import json
import time
import threading
from contextlib import contextmanager
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from collections import Counter, defaultdict
import statistics
import hashlib

# Environment and MCP imports
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

# Firebase imports
import firebase_admin
from firebase_admin import credentials, firestore

# Supabase/PostgreSQL imports
import psycopg2
import psycopg2.extras
from psycopg2.pool import ThreadedConnectionPool

# Visualization imports
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load environment variables
load_dotenv()

# ================================
# FIREBASE CONFIGURATION
# ================================

SERVICE_ACCOUNTS = {
    "default": os.getenv("FIREBASE_CRED_JSON"),
    "project1": os.getenv("FIREBASE_PROJECT1_JSON"),
    # Add more projects here
}

firebase_apps = {}

# Initialize Firebase apps
for name, path in SERVICE_ACCOUNTS.items():
    if name not in firebase_admin._apps and path:
        try:
            cred = credentials.Certificate(path)
            firebase_apps[name] = firebase_admin.initialize_app(cred, name=name)
        except Exception as e:
            print(f"Warning: Failed to initialize Firebase project '{name}': {e}")

def get_firestore(project: str = "default"):
    """Get Firestore client for a specific project"""
    if project not in firebase_apps:
        raise ValueError(f"Firebase project '{project}' is not initialized.")
    return firestore.client(app=firebase_apps[project])

# ================================
# SUPABASE CONFIGURATION
# ================================

# Global connection pool
_connection_pool = None
_pool_lock = threading.Lock()

DATABASE_URL = os.getenv("SUPABASE_CONNECTION_STRING")

def get_connection_pool():
    """Get or create the database connection pool"""
    global _connection_pool
    
    if not DATABASE_URL:
        raise ValueError("SUPABASE_CONNECTION_STRING environment variable is not set")
    
    with _pool_lock:
        if _connection_pool is None:
            _connection_pool = ThreadedConnectionPool(
                minconn=2,
                maxconn=8,
                dsn=DATABASE_URL
            )
    
    return _connection_pool

@contextmanager
def get_db_connection():
    """Context manager for database connections with connection pooling"""
    pool = get_connection_pool()
    conn = None
    try:
        conn = pool.getconn()
        yield conn
    finally:
        if conn:
            pool.putconn(conn)

def execute_safe_query(query: str, params: tuple = None) -> List[Dict[str, Any]]:
    """Execute a read-only query safely"""
    if not is_safe_query(query):
        raise ValueError("Only SELECT, SHOW, DESCRIBE, and EXPLAIN queries are allowed")
    
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

def is_safe_query(query: str) -> bool:
    """Check if a SQL query is safe (read-only)"""
    clean_query = re.sub(r'--.*$', '', query, flags=re.MULTILINE)
    clean_query = re.sub(r'/\*.*?\*/', '', clean_query, flags=re.DOTALL)
    clean_query = ' '.join(clean_query.split()).strip().upper()
    
    safe_patterns = [
        r'^SELECT\s+',
        r'^SHOW\s+',
        r'^DESCRIBE\s+',
        r'^DESC\s+',
        r'^EXPLAIN\s+',
        r'^WITH\s+.*\s+SELECT\s+',
    ]
    
    for pattern in safe_patterns:
        if re.match(pattern, clean_query):
            break
    else:
        return False
    
    dangerous_keywords = [
        'INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER', 'TRUNCATE',
        'GRANT', 'REVOKE', 'COMMIT', 'ROLLBACK', 'SAVEPOINT', 'LOCK'
    ]
    
    for keyword in dangerous_keywords:
        if re.search(r'\b' + keyword + r'\b', clean_query):
            return False
    
    return True

# ================================
# HELPER FUNCTIONS
# ================================

def _analyze_dates(data: List[Dict], date_field: str) -> Dict[str, Any]:
    """Analyze date patterns in the data"""
    dates = [doc.get(date_field) for doc in data if doc.get(date_field)]
    if not dates:
        return {"error": "No valid dates found"}
    
    parsed_dates = []
    for date in dates:
        if isinstance(date, str):
            try:
                parsed_dates.append(datetime.fromisoformat(date.replace('Z', '+00:00')))
            except:
                continue
        elif hasattr(date, 'timestamp'):
            parsed_dates.append(date)
    
    if not parsed_dates:
        return {"error": "Could not parse dates"}
    
    return {
        "earliest": min(parsed_dates).isoformat(),
        "latest": max(parsed_dates).isoformat(),
        "total_span_days": (max(parsed_dates) - min(parsed_dates)).days,
        "total_records": len(parsed_dates)
    }

def _analyze_categorical_field(data: List[Dict], field: str) -> Dict[str, Any]:
    """Analyze distribution of categorical field"""
    values = [doc.get(field, "") for doc in data if doc.get(field)]
    counter = Counter(values)
    
    return {
        "unique_values": len(counter),
        "total_records": len(values),
        "top_values": counter.most_common(10),
        "distribution": dict(counter)
    }

def _analyze_file_names(file_names: List[str]) -> Dict[str, Any]:
    """Analyze file name patterns"""
    extensions = Counter()
    name_lengths = []
    
    for name in file_names:
        if '.' in name:
            ext = name.split('.')[-1].lower()
            extensions[ext] += 1
        name_lengths.append(len(name))
    
    return {
        "total_files": len(file_names),
        "file_extensions": dict(extensions.most_common(10)),
        "name_length_stats": {
            "average": round(statistics.mean(name_lengths), 2) if name_lengths else 0,
            "min": min(name_lengths) if name_lengths else 0,
            "max": max(name_lengths) if name_lengths else 0
        }
    }

def _analyze_folder_structure(folders: List[str]) -> Dict[str, Any]:
    """Analyze folder structure and hierarchy"""
    folder_counter = Counter(folders)
    depths = [folder.count('/') for folder in folders if folder]
    
    return {
        "unique_folders": len(folder_counter),
        "folder_distribution": dict(folder_counter.most_common(10)),
        "depth_analysis": {
            "max_depth": max(depths) if depths else 0,
            "average_depth": round(statistics.mean(depths), 2) if depths else 0
        }
    }

def _analyze_temporal_patterns(dates: List[str]) -> Dict[str, Any]:
    """Analyze temporal patterns in file creation"""
    try:
        parsed_dates = []
        for date in dates:
            if isinstance(date, str):
                try:
                    parsed_dates.append(datetime.fromisoformat(date.replace('Z', '+00:00')))
                except:
                    continue
        
        if not parsed_dates:
            return {"error": "No valid dates to analyze"}
        
        monthly_counts = Counter()
        daily_counts = Counter()
        
        for date in parsed_dates:
            monthly_counts[date.strftime("%Y-%m")] += 1
            daily_counts[date.strftime("%A")] += 1
        
        return {
            "monthly_activity": dict(monthly_counts.most_common(12)),
            "day_of_week_activity": dict(daily_counts),
            "peak_month": monthly_counts.most_common(1)[0] if monthly_counts else None,
            "peak_day": daily_counts.most_common(1)[0] if daily_counts else None
        }
        
    except Exception as e:
        return {"error": f"Temporal analysis failed: {str(e)}"}

def _analyze_collaboration(owners: List[str]) -> Dict[str, Any]:
    """Analyze collaboration patterns"""
    owner_counter = Counter(owners)
    
    return {
        "unique_contributors": len(owner_counter),
        "top_contributors": owner_counter.most_common(10),
        "contribution_distribution": dict(owner_counter),
        "collaboration_score": len(owner_counter) / len(owners) if owners else 0
    }

def _get_date_range(data: List[Dict], date_field: str) -> Dict[str, str]:
    """Get date range from data"""
    dates = [doc.get(date_field) for doc in data if doc.get(date_field)]
    if not dates:
        return {"error": "No dates found"}
    
    return {
        "earliest": min(dates),
        "latest": max(dates)
    }

def _filter_by_date_range(data: List[Dict], date_field: str, start_date: str, end_date: str) -> List[Dict]:
    """Filter data by date range"""
    if not start_date or not end_date:
        return data
    
    filtered = []
    for doc in data:
        doc_date = doc.get(date_field)
        if doc_date and start_date <= doc_date <= end_date:
            filtered.append(doc)
    
    return filtered

def cleanup_connections():
    """Clean up database connection pool on shutdown"""
    global _connection_pool
    with _pool_lock:
        if _connection_pool:
            _connection_pool.closeall()
            _connection_pool = None
    
    with _cache_lock:
        _tool_call_cache.clear()

def _create_call_signature(tool_name: str, **kwargs) -> str:
    """Create a unique signature for a tool call"""
    # Sort kwargs to ensure consistent hashing
    sorted_params = sorted(kwargs.items())
    signature_data = f"{tool_name}:{sorted_params}"
    return hashlib.md5(signature_data.encode()).hexdigest()

# Add this to track first-time calls
_first_calls = set()

def _is_duplicate_call(tool_name: str, **kwargs) -> bool:
    """Check if this tool call is a duplicate of a recent call"""
    signature = _create_call_signature(tool_name, **kwargs)
    
    with _cache_lock:
        current_time = time.time()
        
        # Clean old entries (older than 10 minutes)
        expired_keys = [k for k, v in _tool_call_cache.items() if current_time - v > 600]
        for key in expired_keys:
            del _tool_call_cache[key]
            _first_calls.discard(key)
        
        # Always allow the very first call of any signature
        if signature not in _first_calls:
            _first_calls.add(signature)
            _tool_call_cache[signature] = current_time
            return False
            
        # Check if this call was made recently (within last 2 minutes)
        if signature in _tool_call_cache and current_time - _tool_call_cache[signature] < 120:
            return True
        
        # Record this call
        _tool_call_cache[signature] = current_time
        return False
    
def prevent_duplicate_calls(func):
    """Decorator to prevent duplicate tool calls"""
    import functools
    
    @functools.wraps(func)
    def wrapper(**kwargs):  # Changed from *args, **kwargs to just **kwargs
        tool_name = func.__name__
        
        if _is_duplicate_call(tool_name, **kwargs):
            return {
                "error": "Duplicate call detected. This tool was recently called with the same parameters. Please check your context memory for the previous results instead of making redundant calls."
            }
        
        # Execute the function with keyword arguments only
        result = func(**kwargs)  # Changed from func(*args, **kwargs)
        
        # If the call failed, remove it from cache so it can be retried
        if isinstance(result, dict) and "error" in result:
            signature = _create_call_signature(tool_name, **kwargs)
            with _cache_lock:
                _tool_call_cache.pop(signature, None)
        
        return result
    return wrapper

def clear_tool_cache():
    """Clear the tool call cache"""
    global _tool_call_cache, _first_calls
    with _cache_lock:
        cache_size = len(_tool_call_cache) + len(_first_calls)
        _tool_call_cache.clear()
        _first_calls.clear()
        return cache_size

# ================================
# MCP SERVER INITIALIZATION
# ================================

mcp = FastMCP("unified-firebase-supabase-server", port=8005)

# Tool call cache to prevent redundant calls
_tool_call_cache = {}
_cache_lock = threading.Lock()

# ================================
# FIREBASE TOOLS
# ================================

@mcp.tool()
@prevent_duplicate_calls
def firebase_list_collections(project: str = "default") -> List[str]:
    """List all Firebase collections in the specified project
    
    Args:
        project: Firebase project name (default: 'default')
    
    Returns:
        List of collection names or error message
    """
    try:
        db = get_firestore(project)
        return [c.id for c in db.collections()]
    except Exception as e:
        return [f"Error: {str(e)}"]

@mcp.tool()
@prevent_duplicate_calls
def firebase_get_collection_stats(collection_name: str, project: str = "default") -> Dict[str, Any]:
    """Get comprehensive statistics about a Firebase collection
    
    Args:
        collection_name: Name of the Firebase collection
        project: Firebase project name (default: 'default')
    
    Returns:
        Dictionary containing collection statistics and field analysis
    """
    try:
        db = get_firestore(project)
        docs = list(db.collection(collection_name).stream())
        
        if not docs:
            return {"error": f"Collection '{collection_name}' is empty or doesn't exist"}
        
        total_docs = len(docs)
        data = [doc.to_dict() for doc in docs]
        
        stats = {
            "total_documents": total_docs,
            "collection_name": collection_name,
            "project": project
        }
        
        # Analyze field distribution
        field_stats = defaultdict(lambda: {"count": 0, "types": Counter(), "sample_values": []})
        
        for doc_data in data:
            for field, value in doc_data.items():
                field_info = field_stats[field]
                field_info["count"] += 1
                field_info["types"][type(value).__name__] += 1
                
                if len(field_info["sample_values"]) < 5:
                    field_info["sample_values"].append(str(value)[:100])
        
        stats["field_analysis"] = {}
        for field, info in field_stats.items():
            stats["field_analysis"][field] = {
                "present_in_docs": info["count"],
                "coverage_percentage": round((info["count"] / total_docs) * 100, 2),
                "data_types": dict(info["types"]),
                "sample_values": info["sample_values"]
            }
        
        # Analyze specific common fields
        if "createdAt" in field_stats:
            stats["date_analysis"] = _analyze_dates(data, "createdAt")
        if "folder" in field_stats:
            stats["folder_distribution"] = _analyze_categorical_field(data, "folder")
        if "owner" in field_stats:
            stats["owner_distribution"] = _analyze_categorical_field(data, "owner")
        
        return stats
        
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
@prevent_duplicate_calls
def firebase_list_files_sample(
    collection_name: str, 
    folder: str = "", 
    limit: int = 50,
    offset: int = 0,
    project: str = "default"
) -> Dict[str, Any]:
    """List Firebase files with pagination and filtering
    
    Args:
        collection_name: Name of the Firebase collection
        folder: Filter by folder path (optional)
        limit: Maximum number of files to return (default: 50)
        offset: Number of files to skip (default: 0)
        project: Firebase project name (default: 'default')
    
    Returns:
        Dictionary containing files and pagination info
    """
    try:
        db = get_firestore(project)
        query = db.collection(collection_name)
        
        if folder:
            query = query.where("folder", "==", folder)
        
        all_docs = list(query.stream())
        total_count = len(all_docs)
        
        docs_data = [doc.to_dict() | {"id": doc.id} for doc in all_docs[offset:offset + limit]]
        
        return {
            "files": docs_data,
            "pagination": {
                "total_count": total_count,
                "current_page_size": len(docs_data),
                "offset": offset,
                "limit": limit,
                "has_more": offset + limit < total_count
            },
            "collection_name": collection_name,
            "filter_folder": folder if folder else "all"
        }
        
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
@prevent_duplicate_calls
def firebase_analyze_file_patterns(collection_name: str, project: str = "default") -> Dict[str, Any]:
    """Analyze patterns and trends in Firebase file data
    
    Args:
        collection_name: Name of the Firebase collection
        project: Firebase project name (default: 'default')
    
    Returns:
        Dictionary containing pattern analysis results
    """
    try:
        db = get_firestore(project)
        docs = list(db.collection(collection_name).stream())
        
        if not docs:
            return {"error": f"Collection '{collection_name}' is empty"}
        
        data = [doc.to_dict() for doc in docs]
        
        analysis = {
            "collection_name": collection_name,
            "total_files": len(data)
        }
        
        # File name patterns
        if any("fileName" in doc for doc in data):
            file_names = [doc.get("fileName", "") for doc in data if doc.get("fileName")]
            analysis["file_name_analysis"] = _analyze_file_names(file_names)
        
        # Folder structure analysis
        if any("folder" in doc for doc in data):
            folders = [doc.get("folder", "") for doc in data if doc.get("folder")]
            analysis["folder_structure"] = _analyze_folder_structure(folders)
        
        # Temporal analysis
        if any("createdAt" in doc for doc in data):
            dates = [doc.get("createdAt") for doc in data if doc.get("createdAt")]
            analysis["temporal_patterns"] = _analyze_temporal_patterns(dates)
        
        # Owner/collaboration analysis
        if any("owner" in doc for doc in data):
            owners = [doc.get("owner", "") for doc in data if doc.get("owner")]
            analysis["collaboration_analysis"] = _analyze_collaboration(owners)
        
        return analysis
        
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
@prevent_duplicate_calls
def firebase_get_file_details(collection_name: str, file_id: str, project: str = "default") -> Dict:
    """Get detailed information about a specific Firebase file
    
    Args:
        collection_name: Name of the Firebase collection
        file_id: ID of the file document
        project: Firebase project name (default: 'default')
    
    Returns:
        Dictionary containing file details or error message
    """
    try:
        db = get_firestore(project)
        doc = db.collection(collection_name).document(file_id).get()
        if doc.exists:
            return doc.to_dict() | {"id": doc.id}
        return {"error": f"File with ID {file_id} not found in {collection_name}."}
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
@prevent_duplicate_calls
def firebase_search_and_filter(
    collection_name: str,
    filters: Dict[str, Any] = None,
    search_term: str = "",
    search_field: str = "fileName",
    limit: int = 100,
    project: str = "default"
) -> Dict[str, Any]:
    """Advanced search and filtering in Firebase collections
    
    Args:
        collection_name: Name of the Firebase collection
        filters: Dictionary of field-value pairs to filter by
        search_term: Text to search for in the specified field
        search_field: Field to search in (default: 'fileName')
        limit: Maximum number of results (default: 100)
        project: Firebase project name (default: 'default')
    
    Returns:
        Dictionary containing search results and analysis
    """
    try:
        db = get_firestore(project)
        query = db.collection(collection_name)
        
        # Apply filters
        if filters:
            for field, value in filters.items():
                query = query.where(field, "==", value)
        
        docs = list(query.stream())
        data = [doc.to_dict() | {"id": doc.id} for doc in docs]
        
        # Apply search term filtering in memory
        if search_term and search_field:
            data = [
                doc for doc in data 
                if search_term.lower() in str(doc.get(search_field, "")).lower()
            ]
        
        data = data[:limit]
        
        result = {
            "matches": data,
            "total_matches": len(data),
            "search_criteria": {
                "filters": filters or {},
                "search_term": search_term,
                "search_field": search_field
            }
        }
        
        if data:
            result["quick_analysis"] = {
                "date_range": _get_date_range(data, "createdAt"),
                "unique_folders": len(set(doc.get("folder", "") for doc in data)),
                "unique_owners": len(set(doc.get("owner", "") for doc in data))
            }
        
        return result
        
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
@prevent_duplicate_calls
def firebase_compare_time_periods(
    collection_name: str,
    date_field: str = "createdAt",
    period1_start: str = "",
    period1_end: str = "",
    period2_start: str = "",
    period2_end: str = "",
    project: str = "default"
) -> Dict[str, Any]:
    """Compare Firebase file activity between two time periods
    
    Args:
        collection_name: Name of the Firebase collection
        date_field: Field containing date information (default: 'createdAt')
        period1_start: Start date for first period (ISO format)
        period1_end: End date for first period (ISO format)
        period2_start: Start date for second period (ISO format)
        period2_end: End date for second period (ISO format)
        project: Firebase project name (default: 'default')
    
    Returns:
        Dictionary containing comparison results
    """
    try:
        db = get_firestore(project)
        docs = list(db.collection(collection_name).stream())
        data = [doc.to_dict() for doc in docs]
        
        period1_data = _filter_by_date_range(data, date_field, period1_start, period1_end)
        period2_data = _filter_by_date_range(data, date_field, period2_start, period2_end)
        
        comparison = {
            "period1": {
                "date_range": f"{period1_start} to {period1_end}",
                "count": len(period1_data),
                "folders": len(set(doc.get("folder", "") for doc in period1_data)),
                "owners": len(set(doc.get("owner", "") for doc in period1_data))
            },
            "period2": {
                "date_range": f"{period2_start} to {period2_end}",
                "count": len(period2_data),
                "folders": len(set(doc.get("folder", "") for doc in period2_data)),
                "owners": len(set(doc.get("owner", "") for doc in period2_data))
            }
        }
        
        comparison["changes"] = {
            "file_count_change": len(period2_data) - len(period1_data),
            "folder_count_change": comparison["period2"]["folders"] - comparison["period1"]["folders"],
            "owner_count_change": comparison["period2"]["owners"] - comparison["period1"]["owners"]
        }
        
        return comparison
        
    except Exception as e:
        return {"error": str(e)}

# ================================
# SUPABASE TOOLS
# ================================

@mcp.tool()
@prevent_duplicate_calls
def supabase_execute_query(
    query: str, 
    params: List[Any] = None
) -> Dict[str, Any]:
    """Execute a read-only SQL query on Supabase database
    
    Args:
        query: SQL query to execute (SELECT, SHOW, DESCRIBE, EXPLAIN only)
        params: Query parameters as a list
    
    Returns:
        Dictionary containing query results and metadata
    """
    try:
        if not query or not query.strip():
            return {"error": "Query cannot be empty"}
        
        if not is_safe_query(query):
            return {
                "error": "Only read-only queries are allowed (SELECT, SHOW, DESCRIBE, EXPLAIN)"
            }
        
        clean_query = query.strip()
        if clean_query.upper().startswith('SELECT') and 'LIMIT' not in clean_query.upper():
            actual_limit = min(100, 1000)
            clean_query = clean_query.rstrip(';')
            clean_query += f" LIMIT {actual_limit}"
        
        query_params = tuple(params) if params else None
        
        start_time = time.time()
        results = execute_safe_query(clean_query, query_params)
        execution_time = time.time() - start_time
        
        return {
            "success": True,
            "results": results,
            "metadata": {
                "row_count": len(results),
                "execution_time_seconds": round(execution_time, 3),
                "query": clean_query,
                "parameters": params
            }
        }
        
    except Exception as e:
        return {"error": f"Query execution failed: {str(e)}"}

# ================================
# SHARED VISUALIZATION TOOL
# ================================

@mcp.tool()
@prevent_duplicate_calls
def create_data_visualization(
    data: List[Dict[str, Any]],
    chart_type: str = "bar",
    x_column: str = "x",
    y_column: str = "y",
    title: str = "Data Visualization",
    filename: str = "chart.png",
    width: int = 10,
    height: int = 6,
    save_path: str = "./charts/"
) -> Dict[str, Any]:
    """Create a chart from data and save it as an image locally
    
    Args:
        data: List of dictionaries containing the chart data
        chart_type: Type of chart ('bar', 'line', 'scatter', 'pie', 'histogram')
        x_column: Column name for x-axis data
        y_column: Column name for y-axis data
        title: Chart title
        filename: Name of the output file (with .png extension)
        width: Chart width in inches
        height: Chart height in inches
        save_path: Directory path to save the chart
    
    Returns:
        Dictionary with success status and file path
    """
    try:
        os.makedirs(save_path, exist_ok=True)
        
        df = pd.DataFrame(data)
        
        if chart_type != 'pie' and x_column not in df.columns:
            return {"error": f"Column '{x_column}' not found in data"}
        if y_column not in df.columns:
            return {"error": f"Column '{y_column}' not found in data"}
        
        plt.figure(figsize=(width, height))
        
        if chart_type == "bar":
            plt.bar(df[x_column], df[y_column])
            plt.xlabel(x_column)
            plt.ylabel(y_column)
            
        elif chart_type == "line":
            plt.plot(df[x_column], df[y_column], marker='o')
            plt.xlabel(x_column)
            plt.ylabel(y_column)
            
        elif chart_type == "scatter":
            plt.scatter(df[x_column], df[y_column])
            plt.xlabel(x_column)
            plt.ylabel(y_column)
            
        elif chart_type == "pie":
            plt.pie(df[y_column], labels=df[x_column] if x_column in df.columns else None, autopct='%1.1f%%')
            
        elif chart_type == "histogram":
            plt.hist(df[y_column], bins=20, edgecolor='black')
            plt.xlabel(y_column)
            plt.ylabel("Frequency")
            
        else:
            return {"error": f"Unsupported chart type: {chart_type}"}
        
        plt.title(title)
        plt.tight_layout()
        
        if filename == "chart.png":
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"chart_{timestamp}.png"
        
        if not filename.endswith('.png'):
            filename += '.png'
        
        full_path = os.path.join(save_path, filename)
        
        plt.savefig(full_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return {
            "success": True,
            "message": f"Chart saved successfully",
            "file_path": full_path,
            "filename": filename,
            "chart_type": chart_type,
            "data_points": len(df),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        plt.close()  # Ensure plot is closed even on error
        return {"error": str(e)}
    
@mcp.tool()
def clear_cache() -> Dict[str, Any]:
    """Clear the tool call cache to allow re-running previously called tools
    
    Returns:
        Dictionary with success status and cleared cache size
    """
    try:
        cleared_count = clear_tool_cache()
        return {
            "success": True,
            "message": f"Cache cleared successfully. {cleared_count} cached tool calls removed.",
            "cleared_entries": cleared_count,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"error": f"Failed to clear cache: {str(e)}"}
    
if __name__ == "__main__":
    import atexit
    
    # Register cleanup function
    atexit.register(cleanup_connections)
    
    print("Starting Database Schema and Read-Only Query MCP Server...")
    print(f"Database URL configured: {'Yes' if DATABASE_URL else 'No'}")
    
    try:
        mcp.run(transport="sse")
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
        cleanup_connections()
    except Exception as e:
        print(f"Server error: {e}")
        cleanup_connections()