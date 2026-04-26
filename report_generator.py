#!/usr/bin/env python3
"""
Report Generator - Automated professional mining analysis reports.

Generates HTML reports with:
  - Grade statistics and visualizations
  - Model accuracy comparisons
  - Anomaly summaries
  - Volume calculations
  - Mineralization zone maps
"""

import json
from datetime import datetime


class ReportGenerator:
    """Generate professional HTML reports."""
    
    def __init__(self, project_name="Asst Surveyor AI Analysis"):
        self.project_name = project_name
        self.sections = []
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def add_header_section(self, title, subtitle=""):
        """Add title section."""
        self.sections.append({
            'type': 'header',
            'title': title,
            'subtitle': subtitle
        })
    
    def add_text_section(self, title, content):
        """Add text section."""
        self.sections.append({
            'type': 'text',
            'title': title,
            'content': content
        })
    
    def add_table_section(self, title, data, columns=None):
        """Add table section."""
        self.sections.append({
            'type': 'table',
            'title': title,
            'data': data,
            'columns': columns
        })
    
    def add_statistics_section(self, title, stats_dict):
        """Add statistics display section."""
        self.sections.append({
            'type': 'statistics',
            'title': title,
            'stats': stats_dict
        })
    
    def _build_html(self):
        """Build HTML content."""
        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.project_name}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
            color: #333;
            line-height: 1.6;
        }}
        .container {{
            max-width: 1000px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #1a5490;
            border-bottom: 3px solid #1a5490;
            padding-bottom: 10px;
            margin-top: 0;
        }}
        h2 {{
            color: #2c6ab5;
            margin-top: 30px;
            border-left: 4px solid #2c6ab5;
            padding-left: 10px;
        }}
        h3 {{
            color: #4a8fd8;
            margin-top: 20px;
        }}
        .subtitle {{
            color: #666;
            font-style: italic;
            margin: 10px 0 20px 0;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
            background: white;
        }}
        th {{
            background: #1a5490;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: 600;
        }}
        td {{
            padding: 10px 12px;
            border-bottom: 1px solid #ddd;
        }}
        tr:hover {{
            background: #f9f9f9;
        }}
        .stat-box {{
            display: inline-block;
            background: #f0f4f8;
            padding: 15px 20px;
            margin: 10px 10px 10px 0;
            border-radius: 5px;
            border-left: 4px solid #2c6ab5;
            min-width: 200px;
        }}
        .stat-label {{
            font-size: 0.9em;
            color: #666;
            margin-bottom: 5px;
        }}
        .stat-value {{
            font-size: 1.5em;
            font-weight: bold;
            color: #1a5490;
        }}
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            text-align: center;
            color: #999;
            font-size: 0.9em;
        }}
        .success {{
            background: #d4edda;
            border-left-color: #28a745;
            color: #155724;
            padding: 12px;
            border-radius: 4px;
            margin: 10px 0;
        }}
        .warning {{
            background: #fff3cd;
            border-left-color: #ffc107;
            color: #856404;
            padding: 12px;
            border-radius: 4px;
            margin: 10px 0;
        }}
        .info {{
            background: #d1ecf1;
            border-left-color: #0c5460;
            color: #0c5460;
            padding: 12px;
            border-radius: 4px;
            margin: 10px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
"""
        
        # Add sections
        for section in self.sections:
            if section['type'] == 'header':
                html += f"<h1>{section['title']}</h1>\n"
                if section['subtitle']:
                    html += f"<div class='subtitle'>{section['subtitle']}</div>\n"
            
            elif section['type'] == 'text':
                html += f"<h2>{section['title']}</h2>\n"
                html += f"<p>{section['content']}</p>\n"
            
            elif section['type'] == 'statistics':
                html += f"<h2>{section['title']}</h2>\n"
                for key, value in section['stats'].items():
                    html += f"""
    <div class="stat-box">
        <div class="stat-label">{key}</div>
        <div class="stat-value">{value}</div>
    </div>
"""
            
            elif section['type'] == 'table':
                html += f"<h2>{section['title']}</h2>\n"
                html += "<table>\n"
                
                # Header
                if section['columns']:
                    html += "  <tr>\n"
                    for col in section['columns']:
                        html += f"    <th>{col}</th>\n"
                    html += "  </tr>\n"
                
                # Rows
                if isinstance(section['data'], list):
                    for row in section['data']:
                        html += "  <tr>\n"
                        if isinstance(row, dict):
                            for val in row.values():
                                html += f"    <td>{val}</td>\n"
                        else:
                            for val in row:
                                html += f"    <td>{val}</td>\n"
                        html += "  </tr>\n"
                
                html += "</table>\n"
        
        # Footer
        html += f"""
        <div class="footer">
            <p>Report generated by Asst Surveyor AI on {self.timestamp}</p>
            <p>ML-powered mining survey assistant</p>
        </div>
    </div>
</body>
</html>
"""
        return html
    
    def save_html(self, filename='report.html'):
        """Save report to HTML file."""
        html = self._build_html()
        with open(filename, 'w') as f:
            f.write(html)
        print(f"\nReport saved to {filename}")
        return filename
    
    def save_json(self, filename='report.json'):
        """Save report data to JSON."""
        data = {
            'project': self.project_name,
            'timestamp': self.timestamp,
            'sections': self.sections
        }
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        print(f"Report data saved to {filename}")
        return filename
