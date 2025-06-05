import os
from typing import List, Dict, Optional, Any
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
import psycopg2
import psycopg2.extras
from psycopg2.pool import ThreadedConnectionPool
from datetime import datetime, timedelta
import json
import statistics
from collections import Counter, defaultdict
import threading
from contextlib import contextmanager
import time
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load env vars
load_dotenv()

# Global connection pools for each project
_connection_pools = {}
_pool_lock = threading.Lock()

# Load multiple Supabase connection strings
SUPABASE_CONNECTIONS = {
    "default": os.getenv("SUPABASE_CONNECTION_STRING"),
    # "project1": os.getenv("SUPABASE_PROJECT1_CONNECTION_STRING"),
    # Add more projects here
}

def get_connection_pool(project: str = "default"):
    """Get or create a connection pool for the specified project"""
    if project not in SUPABASE_CONNECTIONS:
        raise ValueError(f"Project '{project}' is not configured.")
    
    connection_string = SUPABASE_CONNECTIONS[project]
    if not connection_string:
        raise ValueError(f"Connection string for project '{project}' is not set.")
    
    with _pool_lock:
        if project not in _connection_pools:
            # Create connection pool with 2-10 connections
            _connection_pools[project] = ThreadedConnectionPool(
                minconn=2,
                maxconn=10,
                dsn=connection_string
            )
    
    return _connection_pools[project]

@contextmanager
def get_db_connection(project: str = "default"):
    """Context manager for database connections with connection pooling"""
    pool = get_connection_pool(project)
    conn = None
    try:
        conn = pool.getconn()
        yield conn
    finally:
        if conn:
            pool.putconn(conn)

def execute_query(query: str, params: tuple = None, project: str = "default", fetch: bool = True):
    """Execute a query with optimized connection handling"""
    with get_db_connection(project) as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            cursor.execute(query, params)
            
            if fetch:
                return cursor.fetchall()
            else:
                conn.commit()
                return cursor.rowcount

def execute_query_single(query: str, params: tuple = None, project: str = "default"):
    """Execute a query and return single result - optimized for single row queries"""
    with get_db_connection(project) as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            cursor.execute(query, params)
            return cursor.fetchone()

# MCP Server
mcp = FastMCP("supabase-files-analytics", port=8004)

# Cache for table schemas to avoid repeated queries
_schema_cache = {}
_cache_lock = threading.Lock()

def get_table_schema(table_name: str, project: str = "default"):
    """Get table schema with caching"""
    cache_key = f"{project}:{table_name}"
    
    with _cache_lock:
        if cache_key in _schema_cache:
            return _schema_cache[cache_key]
    
    query = """
    SELECT column_name, data_type, is_nullable, column_default
    FROM information_schema.columns 
    WHERE table_name = %s AND table_schema = 'public'
    ORDER BY ordinal_position;
    """
    
    schema = execute_query(query, (table_name,), project)
    
    with _cache_lock:
        _schema_cache[cache_key] = schema
    
    return schema

@mcp.tool()
def supabase_list_tables(project: str = "default") -> List[str]:
    """List all tables in the Supabase project"""
    try:
        query = """
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public' 
        AND table_type = 'BASE TABLE'
        ORDER BY table_name;
        """
        result = execute_query(query, project=project)
        return [row['table_name'] for row in result]
    except Exception as e:
        return [f"Error: {str(e)}"]

@mcp.tool()
def supabase_get_table_analytics(table_name: str, project: str = "default") -> Dict[str, Any]:
    """Get comprehensive analytics about a Supabase table with optimized queries"""
    try:
        # Get cached schema
        columns = get_table_schema(table_name, project)
        
        if not columns:
            return {"error": f"Table {table_name} not found."}
        
        # Get row count with a single query
        count_result = execute_query_single(f'SELECT COUNT(*) as total_rows FROM "{table_name}";', project=project)
        total_rows = count_result['total_rows'] if count_result else 0
        
        # Build a single comprehensive query for all numeric columns
        numeric_columns = [col['column_name'] for col in columns 
                          if col['data_type'] in ['integer', 'bigint', 'numeric', 'real', 'double precision']]
        
        text_columns = [col['column_name'] for col in columns 
                       if col['data_type'] not in ['integer', 'bigint', 'numeric', 'real', 'double precision']]
        
        column_analysis = {}
        
        # Analyze numeric columns in batches
        if numeric_columns:
            numeric_stats = _analyze_numeric_columns_batch(table_name, numeric_columns, total_rows, project)
            column_analysis.update(numeric_stats)
        
        # Analyze text columns in batches
        if text_columns:
            text_stats = _analyze_text_columns_batch(table_name, text_columns, total_rows, project)
            column_analysis.update(text_stats)
        
        # Add column metadata
        for col in columns:
            col_name = col['column_name']
            if col_name in column_analysis:
                column_analysis[col_name].update({
                    "type": col['data_type'],
                    "nullable": col['is_nullable'] == 'YES'
                })
        
        return {
            "table_name": table_name,
            "project": project,
            "total_rows": total_rows,
            "total_columns": len(columns),
            "column_analysis": column_analysis
        }
        
    except Exception as e:
        return {"error": str(e)}

def _analyze_numeric_columns_batch(table_name: str, columns: List[str], total_rows: int, project: str) -> Dict[str, Any]:
    """Analyze multiple numeric columns in a single query"""
    if not columns:
        return {}
    
    # Build single query for all numeric columns
    select_parts = []
    for col in columns:
        select_parts.extend([
            f'COUNT("{col}") as {col}_non_null_count',
            f'MIN("{col}") as {col}_min_val',
            f'MAX("{col}") as {col}_max_val',
            f'AVG("{col}")::numeric(10,2) as {col}_avg_val'
        ])
    
    query = f'SELECT {", ".join(select_parts)} FROM "{table_name}";'
    result = execute_query_single(query, project=project)
    
    analysis = {}
    for col in columns:
        non_null_count = result[f'{col}_non_null_count'] or 0
        null_percentage = round(((total_rows - non_null_count) / total_rows * 100), 2) if total_rows > 0 else 0
        
        analysis[col] = {
            "total_count": total_rows,
            "non_null_count": non_null_count,
            "null_percentage": null_percentage,
            "min": float(result[f'{col}_min_val']) if result[f'{col}_min_val'] is not None else None,
            "max": float(result[f'{col}_max_val']) if result[f'{col}_max_val'] is not None else None,
            "average": float(result[f'{col}_avg_val']) if result[f'{col}_avg_val'] is not None else None
        }
    
    return analysis

def _analyze_text_columns_batch(table_name: str, columns: List[str], total_rows: int, project: str) -> Dict[str, Any]:
    """Analyze multiple text columns efficiently"""
    if not columns:
        return {}
    
    analysis = {}
    
    # Build single query for basic stats
    select_parts = []
    for col in columns:
        select_parts.extend([
            f'COUNT("{col}") as {col}_non_null_count',
            f'COUNT(DISTINCT "{col}") as {col}_unique_count'
        ])
    
    query = f'SELECT {", ".join(select_parts)} FROM "{table_name}";'
    result = execute_query_single(query, project=project)
    
    for col in columns:
        non_null_count = result[f'{col}_non_null_count'] or 0
        unique_count = result[f'{col}_unique_count'] or 0
        null_percentage = round(((total_rows - non_null_count) / total_rows * 100), 2) if total_rows > 0 else 0
        
        analysis[col] = {
            "total_count": total_rows,
            "non_null_count": non_null_count,
            "null_percentage": null_percentage,
            "unique_count": unique_count,
            "uniqueness_ratio": round(unique_count / non_null_count, 3) if non_null_count > 0 else 0,
            "sample_values": []  # We'll get samples separately if needed
        }
    
    # Get sample values for text columns (only if needed)
    for col in columns[:3]:  # Limit to first 3 columns to avoid too many queries
        if analysis[col]["unique_count"] < 1000:  # Only get samples for columns with reasonable unique count
            sample_query = f'SELECT DISTINCT "{col}" FROM "{table_name}" WHERE "{col}" IS NOT NULL LIMIT 3;'
            samples = execute_query(sample_query, project=project)
            analysis[col]["sample_values"] = [str(row[col])[:50] for row in samples if row[col] is not None]
    
    return analysis

@mcp.tool()
def supabase_list_files_paginated(
    table_name: str, 
    folder: str = "", 
    limit: int = 50,
    offset: int = 0,
    order_by: str = "createdAt",
    order_direction: str = "DESC",
    project: str = "default"
) -> Dict[str, Any]:
    """List Supabase files with pagination and ordering - optimized version"""
    try:
        # Validate limit to prevent excessive queries
        limit = min(limit, 500)  # Cap at 500 rows max
        
        # Build the query with optimized WHERE clause
        where_clause = f'WHERE folder = %s' if folder else ""
        params = (folder,) if folder else ()
        
        # Use a single query to get both count and data when possible
        if offset == 0 and limit <= 100:
            # For small first-page requests, get data + count in one go
            data_query = f'''
            SELECT *, COUNT(*) OVER() as total_count
            FROM "{table_name}" 
            {where_clause}
            ORDER BY "{order_by}" {order_direction}
            LIMIT %s;
            '''
            
            data_params = params + (limit,)
            result = execute_query(data_query, data_params, project)
            
            total_count = result[0]['total_count'] if result else 0
            # Remove the total_count field from each row
            for row in result:
                del row['total_count']
            
        else:
            # For larger offsets, use separate queries
            count_query = f'SELECT COUNT(*) as total FROM "{table_name}" {where_clause};'
            total_result = execute_query_single(count_query, params, project)
            total_count = total_result['total'] if total_result else 0
            
            data_query = f'''
            SELECT * FROM "{table_name}" 
            {where_clause}
            ORDER BY "{order_by}" {order_direction}
            LIMIT %s OFFSET %s;
            '''
            
            data_params = params + (limit, offset)
            result = execute_query(data_query, data_params, project)
        
        return {
            "files": [dict(row) for row in result],
            "pagination": {
                "total_count": total_count,
                "current_page_size": len(result),
                "offset": offset,
                "limit": limit,
                "has_more": offset + limit < total_count,
                "total_pages": (total_count + limit - 1) // limit
            },
            "table_name": table_name,
            "filter_folder": folder if folder else "all",
            "ordering": f"{order_by} {order_direction}"
        }
        
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
def supabase_analyze_data_patterns(table_name: str, project: str = "default", quick_mode: bool = False) -> Dict[str, Any]:
    """Analyze patterns and distributions with performance optimizations"""
    try:
        analysis = {
            "table_name": table_name,
            "project": project
        }
        
        if quick_mode:
            # Quick mode - use smaller limits and fewer queries
            folder_limit = 10
            owner_limit = 10
            ext_limit = 10
            temporal_limit = 12
        else:
            folder_limit = 20
            owner_limit = 15
            ext_limit = 15
            temporal_limit = 24
        
        # Use a single query to get multiple distributions
        combined_query = f'''
        WITH folder_stats AS (
            SELECT 'folder' as type, folder as value, COUNT(*) as count 
            FROM "{table_name}" 
            WHERE folder IS NOT NULL AND folder != ''
            GROUP BY folder 
            ORDER BY count DESC 
            LIMIT {folder_limit}
        ),
        owner_stats AS (
            SELECT 'owner' as type, owner as value, COUNT(*) as count 
            FROM "{table_name}" 
            WHERE owner IS NOT NULL AND owner != ''
            GROUP BY owner 
            ORDER BY count DESC 
            LIMIT {owner_limit}
        )
        SELECT * FROM folder_stats
        UNION ALL
        SELECT * FROM owner_stats;
        '''
        
        combined_result = execute_query(combined_query, project=project)
        
        # Parse combined results
        folder_results = [row for row in combined_result if row['type'] == 'folder']
        owner_results = [row for row in combined_result if row['type'] == 'owner']
        
        analysis["folder_distribution"] = {
            "total_folders": len(folder_results),
            "top_folders": [{"folder": row['value'], "count": row['count']} for row in folder_results]
        }
        
        analysis["owner_distribution"] = {
            "total_owners": len(owner_results),
            "top_owners": [{"owner": row['value'], "count": row['count']} for row in owner_results]
        }
        
        # File extensions analysis - only if fileName column exists
        try:
            ext_query = f'''
            SELECT 
                CASE 
                    WHEN "fileName" LIKE '%.%' 
                    THEN LOWER(SUBSTRING("fileName" FROM '\.([^.]*)$'))
                    ELSE 'no_extension'
                END as extension,
                COUNT(*) as count
            FROM "{table_name}"
            WHERE "fileName" IS NOT NULL AND "fileName" != ''
            GROUP BY extension
            ORDER BY count DESC
            LIMIT {ext_limit};
            '''
            ext_result = execute_query(ext_query, project=project)
            analysis["file_extensions"] = [{"extension": row['extension'], "count": row['count']} for row in ext_result]
        except:
            analysis["file_extensions"] = []
        
        # Temporal analysis - only if createdAt exists
        try:
            temporal_query = f'''
            SELECT 
                DATE_TRUNC('month', "createdAt"::timestamp) as month,
                COUNT(*) as count
            FROM "{table_name}"
            WHERE "createdAt" IS NOT NULL
            GROUP BY month
            ORDER BY month DESC
            LIMIT {temporal_limit};
            '''
            temporal_result = execute_query(temporal_query, project=project)
            analysis["temporal_patterns"] = {
                "monthly_activity": [{"month": row['month'].strftime('%Y-%m'), "count": row['count']} for row in temporal_result]
            }
        except:
            analysis["temporal_patterns"] = {"monthly_activity": []}
        
        return analysis
        
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
def supabase_get_file_details(table_name: str, file_id: str, project: str = "default") -> Dict:
    """Get detailed information about a specific file - optimized single query"""
    try:
        result = execute_query_single(f'SELECT * FROM "{table_name}" WHERE id = %s;', (file_id,), project)
        
        if result:
            return dict(result)
        return {"error": f"File with ID {file_id} not found in {table_name}."}
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
def supabase_search_and_filter(
    table_name: str,
    filters: Dict[str, Any] = None,
    search_term: str = "",
    search_field: str = "fileName",
    limit: int = 100,
    project: str = "default"
) -> Dict[str, Any]:
    """Advanced search and filtering with optimized query building"""
    try:
        # Cap limit for performance
        limit = min(limit, 500)
        
        # Build WHERE clause efficiently
        where_conditions = []
        params = []
        
        if filters:
            for field, value in filters.items():
                where_conditions.append(f'"{field}" = %s')
                params.append(value)
        
        if search_term and search_field:
            where_conditions.append(f'"{search_field}" ILIKE %s')
            params.append(f"%{search_term}%")
        
        where_clause = f"WHERE {' AND '.join(where_conditions)}" if where_conditions else ""
        
        # Use index-friendly ordering
        query = f'''
        SELECT * FROM "{table_name}" 
        {where_clause}
        ORDER BY "createdAt" DESC
        LIMIT %s;
        '''
        params.append(limit)
        
        result = execute_query(query, tuple(params), project)
        data = [dict(row) for row in result]
        
        analysis_result = {
            "matches": data,
            "total_matches": len(data),
            "search_criteria": {
                "filters": filters or {},
                "search_term": search_term,
                "search_field": search_field
            }
        }
        
        # Quick analysis of results (optimized)
        if data:
            folders = {row.get("folder", "") for row in data if row.get("folder")}
            owners = {row.get("owner", "") for row in data if row.get("owner")}
            
            analysis_result["quick_analysis"] = {
                "unique_folders": len(folders),
                "unique_owners": len(owners),
                "date_range": _get_date_range_optimized(data)
            }
        
        return analysis_result
        
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
def supabase_compare_time_periods(
    table_name: str,
    date_field: str = "createdAt",
    period1_start: str = "",
    period1_end: str = "",
    period2_start: str = "",
    period2_end: str = "",
    project: str = "default"
) -> Dict[str, Any]:
    """Compare time periods with a single optimized query"""
    try:
        # Single query for both periods
        query = f'''
        SELECT 
            CASE 
                WHEN "{date_field}" >= %s AND "{date_field}" <= %s THEN 'period1'
                WHEN "{date_field}" >= %s AND "{date_field}" <= %s THEN 'period2'
                ELSE 'other'
            END as period,
            COUNT(*) as count,
            COUNT(DISTINCT folder) as folders,
            COUNT(DISTINCT owner) as owners
        FROM "{table_name}"
        WHERE ("{date_field}" >= %s AND "{date_field}" <= %s) 
           OR ("{date_field}" >= %s AND "{date_field}" <= %s)
        GROUP BY period;
        '''
        
        params = (period1_start, period1_end, period2_start, period2_end,
                 period1_start, period1_end, period2_start, period2_end)
        
        result = execute_query(query, params, project)
        
        # Parse results
        period1_data = next((row for row in result if row['period'] == 'period1'), 
                           {'count': 0, 'folders': 0, 'owners': 0})
        period2_data = next((row for row in result if row['period'] == 'period2'), 
                           {'count': 0, 'folders': 0, 'owners': 0})
        
        comparison = {
            "period1": {
                "date_range": f"{period1_start} to {period1_end}",
                "count": period1_data['count'],
                "folders": period1_data['folders'],
                "owners": period1_data['owners']
            },
            "period2": {
                "date_range": f"{period2_start} to {period2_end}",
                "count": period2_data['count'],
                "folders": period2_data['folders'],
                "owners": period2_data['owners']
            }
        }
        
        # Calculate changes
        comparison["changes"] = {
            "file_count_change": period2_data['count'] - period1_data['count'],
            "folder_count_change": period2_data['folders'] - period1_data['folders'],
            "owner_count_change": period2_data['owners'] - period1_data['owners']
        }
        
        return comparison
        
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
def supabase_aggregate_analysis(
    table_name: str,
    group_by_field: str,
    aggregation_field: str = "id",
    aggregation_type: str = "count",
    limit: int = 20,
    project: str = "default"
) -> Dict[str, Any]:
    """Perform aggregation analysis with optimized query"""
    try:
        # Validate aggregation type
        valid_aggregations = ["count", "sum", "avg", "min", "max"]
        if aggregation_type not in valid_aggregations:
            return {"error": f"Invalid aggregation type. Use one of: {valid_aggregations}"}
        
        # Cap limit for performance
        limit = min(limit, 100)
        
        # Build aggregation query
        if aggregation_type == "count":
            agg_expr = f'COUNT("{aggregation_field}") as value'
        else:
            agg_expr = f'{aggregation_type.upper()}("{aggregation_field}") as value'
        
        query = f'''
        SELECT "{group_by_field}" as group_key, {agg_expr}
        FROM "{table_name}"
        WHERE "{group_by_field}" IS NOT NULL
        GROUP BY "{group_by_field}"
        ORDER BY value DESC
        LIMIT %s;
        '''
        
        result = execute_query(query, (limit,), project)
        
        return {
            "table_name": table_name,
            "aggregation": {
                "group_by": group_by_field,
                "field": aggregation_field,
                "type": aggregation_type
            },
            "results": [{"group": row['group_key'], "value": row['value']} for row in result],
            "total_groups": len(result)
        }
        
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
def supabase_create_chart(
    data: List[Dict[str, Any]],
    chart_type: str = "bar",
    x_column: str = "x",
    y_column: str = "y",
    title: str = "Chart",
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

        # Create save directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)
        
        # Convert data to DataFrame
        df = pd.DataFrame(data)
        
        # Validate required columns exist
        if chart_type != 'pie' and x_column not in df.columns:
            return {"error": f"Column '{x_column}' not found in data"}
        if y_column not in df.columns:
            return {"error": f"Column '{y_column}' not found in data"}
        
        # Create figure and axis
        plt.figure(figsize=(width, height))
        
        # Create chart based on type
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
        
        # Set title and improve layout
        plt.title(title)
        plt.tight_layout()
        
        # Add timestamp to filename if not specified
        if filename == "chart.png":
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"chart_{timestamp}.png"
        
        # Ensure filename has .png extension
        if not filename.endswith('.png'):
            filename += '.png'
        
        # Full file path
        full_path = os.path.join(save_path, filename)
        
        # Save the chart
        plt.savefig(full_path, dpi=300, bbox_inches='tight')
        plt.close()  # Close to free memory
        
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

# Optimized helper functions
def _get_date_range_optimized(data: List[Dict]) -> Dict[str, str]:
    """Get date range with optimized processing"""
    dates = []
    date_fields = ['createdAt', 'created_at', 'updatedAt', 'updated_at']
    
    for row in data:
        for field in date_fields:
            if field in row and row[field]:
                dates.append(str(row[field]))
                break  # Only take first available date field
    
    if not dates:
        return {"error": "No dates found"}
    
    # Use min/max directly instead of sorting
    return {
        "earliest": min(dates),
        "latest": max(dates),
        "count": len(dates)
    }

# Add cleanup function for connection pools
def cleanup_connections():
    """Clean up connection pools on shutdown"""
    with _pool_lock:
        for pool in _connection_pools.values():
            pool.closeall()
        _connection_pools.clear()

# Resources remain the same but will benefit from faster queries
@mcp.resource("supabase-files://{project}/{table_name}/analytics")
def get_supabase_table_analytics_resource(project: str, table_name: str) -> str:
    """Get Supabase table analytics as a formatted resource"""
    analytics = supabase_get_table_analytics(table_name, project)
    if "error" in analytics:
        return f"# Error\n\n{analytics['error']}"
    
    content = f"# Supabase Table Analytics: {table_name}\n\n"
    content += f"**Project**: {project}\n"
    content += f"**Total Rows**: {analytics['total_rows']}\n"
    content += f"**Total Columns**: {analytics['total_columns']}\n\n"
    
    content += "## Column Analysis\n\n"
    for col_name, info in analytics.get('column_analysis', {}).items():
        content += f"### {col_name}\n"
        content += f"- Type: {info['type']}\n"
        content += f"- Nullable: {info['nullable']}\n"
        content += f"- Null Rate: {info['null_percentage']}%\n"
        
        if 'unique_count' in info:
            content += f"- Unique Values: {info['unique_count']} (ratio: {info['uniqueness_ratio']})\n"
            if info['sample_values']:
                content += f"- Sample Values: {', '.join(info['sample_values'][:3])}\n"
        
        if 'min' in info and info['min'] is not None:
            content += f"- Range: {info['min']} to {info['max']} (avg: {info['average']})\n"
        
        content += "\n"
    
    return content

@mcp.resource("supabase-files://{project}/{table_name}/patterns")
def get_supabase_patterns_resource(project: str, table_name: str) -> str:
    """Get Supabase data patterns as a formatted resource"""
    patterns = supabase_analyze_data_patterns(table_name, project, quick_mode=True)
    if "error" in patterns:
        return f"# Error\n\n{patterns['error']}"
    
    content = f"# Data Patterns: {table_name}\n\n"
    
    if 'folder_distribution' in patterns:
        content += "## Folder Distribution\n"
        for folder in patterns['folder_distribution']['top_folders'][:10]:
            content += f"- {folder['folder']}: {folder['count']} files\n"
        content += "\n"
    
    if 'owner_distribution' in patterns:
        content += "## Owner Distribution\n"
        for owner in patterns['owner_distribution']['top_owners'][:10]:
            content += f"- {owner['owner']}: {owner['count']} files\n"
        content += "\n"
    
    if 'file_extensions' in patterns:
        content += "## File Extensions\n"
        for ext in patterns['file_extensions'][:10]:
            content += f"- .{ext['extension']}: {ext['count']} files\n"
        content += "\n"
    
    return content

@mcp.prompt()
def generate_supabase_table_analysis_prompt(project: str, table_name: str) -> str:
    """Generate comprehensive analysis prompt for a Supabase table"""
    return f"""
    Analyze the Supabase table '{table_name}' in project '{project}' using these optimized tools:
    
    1. Use `supabase_get_table_analytics` to get overall statistics and column analysis
    2. Use `supabase_analyze_data_patterns` with quick_mode=True for faster pattern analysis
    3. Use `supabase_list_files_paginated` with reasonable limits (50-100) for data samples
    4. Use `supabase_search_and_filter` for targeted data investigation
    5. Use `supabase_aggregate_analysis` for specific grouping insights
    
    Provide a comprehensive analysis including:
    - Data quality assessment
    - Usage patterns and trends
    - Column-level insights
    - Organizational structure analysis
    - Performance recommendations
    - Key insights and anomalies
    
    Note: All tools are now optimized for better performance with connection pooling,
    query batching, and reduced database round trips.
    """

if __name__ == "__main__":
    import atexit
    atexit.register(cleanup_connections)
    mcp.run(transport="sse")