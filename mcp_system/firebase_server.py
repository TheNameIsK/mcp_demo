import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime, timedelta
from collections import Counter, defaultdict
import statistics
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load env vars
load_dotenv()

# Load multiple service accounts
SERVICE_ACCOUNTS = {
    "default": os.getenv("FIREBASE_CRED_JSON"),
    "project1": os.getenv("FIREBASE_PROJECT1_JSON"),
    # Add more projects here
}

firebase_apps = {}

for name, path in SERVICE_ACCOUNTS.items():
    if name not in firebase_admin._apps and path:
        cred = credentials.Certificate(path)
        firebase_apps[name] = firebase_admin.initialize_app(cred, name=name)

def get_firestore(project: str = "default"):
    if project not in firebase_apps:
        raise ValueError(f"Project '{project}' is not initialized.")
    return firestore.client(app=firebase_apps[project])

# MCP Server
mcp = FastMCP("firebase-files-analytics", port=8003)

@mcp.tool()
def firebase_list_collections(project: str = "default") -> List[str]:
    """List all Firebase collections in the project"""
    try:
        db = get_firestore(project)
        return [c.id for c in db.collections()]
    except Exception as e:
        return [f"Error: {str(e)}"]

@mcp.tool()
def firebase_get_collection_stats(collection_name: str, project: str = "default") -> Dict[str, Any]:
    """Get comprehensive statistics about a Firebase collection"""
    try:
        db = get_firestore(project)
        docs = list(db.collection(collection_name).stream())
        
        if not docs:
            return {"error": f"Collection '{collection_name}' is empty or doesn't exist"}
        
        total_docs = len(docs)
        data = [doc.to_dict() for doc in docs]
        
        # Basic stats
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
                
                # Store sample values (limit to 5)
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
def firebase_list_files_sample(
    collection_name: str, 
    folder: str = "", 
    limit: int = 50,
    offset: int = 0,
    project: str = "default"
) -> Dict[str, Any]:
    """List Firebase files with pagination and smart sampling for analysis"""
    try:
        db = get_firestore(project)
        query = db.collection(collection_name)
        
        if folder:
            query = query.where("folder", "==", folder)
        
        # Get total count first
        all_docs = list(query.stream())
        total_count = len(all_docs)
        
        # Apply pagination
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
def firebase_analyze_file_patterns(collection_name: str, project: str = "default") -> Dict[str, Any]:
    """Analyze patterns and trends in Firebase file data"""
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
def firebase_get_file_details(collection_name: str, file_id: str, project: str = "default") -> Dict:
    """Get detailed information about a specific Firebase file"""
    try:
        db = get_firestore(project)
        doc = db.collection(collection_name).document(file_id).get()
        if doc.exists:
            return doc.to_dict() | {"id": doc.id}
        return {"error": f"File with ID {file_id} not found in {collection_name}."}
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
def firebase_search_and_filter(
    collection_name: str,
    filters: Dict[str, Any] = None,
    search_term: str = "",
    search_field: str = "fileName",
    limit: int = 100,
    project: str = "default"
) -> Dict[str, Any]:
    """Advanced search and filtering in Firebase collections with analysis context"""
    try:
        db = get_firestore(project)
        query = db.collection(collection_name)
        
        # Apply filters
        if filters:
            for field, value in filters.items():
                query = query.where(field, "==", value)
        
        docs = list(query.stream())
        data = [doc.to_dict() | {"id": doc.id} for doc in docs]
        
        # Apply search term filtering in memory (Firestore has limited text search)
        if search_term and search_field:
            data = [
                doc for doc in data 
                if search_term.lower() in str(doc.get(search_field, "")).lower()
            ]
        
        # Apply limit
        data = data[:limit]
        
        # Provide context for analysis
        result = {
            "matches": data,
            "total_matches": len(data),
            "search_criteria": {
                "filters": filters or {},
                "search_term": search_term,
                "search_field": search_field
            }
        }
        
        # Quick analysis of results
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
def firebase_compare_time_periods(
    collection_name: str,
    date_field: str = "createdAt",
    period1_start: str = "",
    period1_end: str = "",
    period2_start: str = "",
    period2_end: str = "",
    project: str = "default"
) -> Dict[str, Any]:
    """Compare Firebase file activity between two time periods"""
    try:
        db = get_firestore(project)
        docs = list(db.collection(collection_name).stream())
        data = [doc.to_dict() for doc in docs]
        
        # Filter data by periods
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
        
        # Calculate changes
        comparison["changes"] = {
            "file_count_change": len(period2_data) - len(period1_data),
            "folder_count_change": comparison["period2"]["folders"] - comparison["period1"]["folders"],
            "owner_count_change": comparison["period2"]["owners"] - comparison["period1"]["owners"]
        }
        
        return comparison
        
    except Exception as e:
        return {"error": str(e)}
    
@mcp.tool()
def firebase_create_chart(
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

# Helper function
def _analyze_dates(data: List[Dict], date_field: str) -> Dict[str, Any]:
    """Analyze date patterns in the data"""
    dates = [doc.get(date_field) for doc in data if doc.get(date_field)]
    if not dates:
        return {"error": "No valid dates found"}
    
    # Convert to datetime objects if they're strings
    parsed_dates = []
    for date in dates:
        if isinstance(date, str):
            try:
                parsed_dates.append(datetime.fromisoformat(date.replace('Z', '+00:00')))
            except:
                continue
        elif hasattr(date, 'timestamp'):  # Firestore timestamp
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
    
    # Analyze folder depth (count '/' separators)
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
        
        # Group by month and day of week
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

@mcp.resource("firebase-files://{project}/{collection_name}/stats")
def get_firebase_collection_stats_resource(project: str, collection_name: str) -> str:
    """Get collection statistics as a formatted resource"""
    stats = firebase_get_collection_stats(collection_name, project)
    if "error" in stats:
        return f"# Error\n\n{stats['error']}"
    
    content = f"# Firebase Collection Analytics: {collection_name}\n\n"
    content += f"**Project**: {project}\n"
    content += f"**Total Documents**: {stats['total_documents']}\n\n"
    
    content += "## Field Analysis\n\n"
    for field, info in stats.get('field_analysis', {}).items():
        content += f"### {field}\n"
        content += f"- Coverage: {info['coverage_percentage']}% ({info['present_in_docs']}/{stats['total_documents']} docs)\n"
        content += f"- Data Types: {', '.join(info['data_types'].keys())}\n"
        content += f"- Sample Values: {', '.join(info['sample_values'][:3])}\n\n"
    
    if 'date_analysis' in stats:
        content += "## Date Analysis\n"
        date_info = stats['date_analysis']
        content += f"- Date Range: {date_info.get('earliest', 'N/A')} to {date_info.get('latest', 'N/A')}\n"
        content += f"- Total Span: {date_info.get('total_span_days', 0)} days\n\n"
    
    return content

@mcp.prompt()
def generate_firebase_collection_analysis_prompt(project: str, collection_name: str) -> str:
    """Generate comprehensive analysis prompt for a Firebase collection"""
    return f"""
    Analyze the Firebase collection '{collection_name}' in project '{project}' using these tools:
    
    1. Use `firebase_get_collection_stats` to get overall statistics and field distribution
    2. Use `firebase_analyze_file_patterns` to identify patterns and trends
    3. Use `firebase_list_files_sample` with different parameters to examine data samples
    4. Use `firebase_search_and_filter` to investigate specific data segments
    
    Provide a comprehensive analysis including:
    - Data quality assessment
    - Usage patterns and trends
    - Organizational structure analysis
    - Recommendations for data optimization
    - Key insights and anomalies
    """

if __name__ == "__main__":
    mcp.run(transport="sse")