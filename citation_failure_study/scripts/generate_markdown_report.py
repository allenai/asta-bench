#!/usr/bin/env python3
"""
Generate a comprehensive markdown report from citation and claim failure mode analysis results.
Combines data from both JSON reports into a single readable markdown document.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple


def load_json_report(filepath: Path) -> Dict[str, Any]:
    """Load a JSON report file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def format_percentage(value: float) -> str:
    """Format a percentage value for display."""
    return f"{value:.2f}%"


def create_markdown_table(headers: List[str], rows: List[List[str]]) -> str:
    """Create a markdown table from headers and rows."""
    table = []
    table.append("| " + " | ".join(headers) + " |")
    table.append("|" + "|".join(["---" for _ in headers]) + "|")
    for row in rows:
        table.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(table)


def generate_executive_summary(claims_report: Dict, citations_report: Dict, totals: Dict) -> str:
    """Generate the executive summary section."""
    summary = []
    summary.append("## Executive Summary\n")
    
    # Key metrics
    total_sampled = claims_report.get("sample_size", 200)
    total_claims_analyzed = claims_report.get("total_claims_classified", 0)
    missing_citations = total_sampled - total_claims_analyzed
    total_citations = citations_report.get("total_citations_analyzed", 0)
    
    summary.append(f"This report analyzes failure modes in scientific question-answering systems based on:")
    summary.append(f"- **{total_sampled:,}** unsupported claims sampled (50 per system)")
    summary.append(f"- **{total_claims_analyzed:,}** claims with citations analyzed")
    summary.append(f"- **{missing_citations:,}** claims without any citations")
    summary.append(f"- **{total_citations:,}** non-supporting citations analyzed")
    
    summary.append("")
    
    # Top failure modes - adjusted for total sampled
    if claims_report.get("failure_mode_distribution"):
        # Check if "No Citation Provided" would be the top mode
        no_citation_pct = (missing_citations / total_sampled) * 100 if total_sampled > 0 else 0
        
        top_claim_mode = max(claims_report["failure_mode_distribution"].items(), 
                            key=lambda x: x[1]["count"])
        top_claim_count = top_claim_mode[1]["count"]
        # Calculate percentage of ALL claims (using totals from extract_totals)
        grand_totals = totals.get('grand_totals', {})
        # Since we only analyzed a sample, we need to use the overall failure rate
        # multiplied by the distribution within failures
        overall_failure_rate = grand_totals.get('claim_failure_rate', 0) / 100
        # Percentage of this failure mode among all failures
        pct_of_failures = top_claim_mode[1]['percentage'] / 100
        # Estimated percentage of all claims
        top_claim_pct = overall_failure_rate * pct_of_failures * 100
        
        if missing_citations > top_claim_count:
            # Estimate percentage of all claims that have no citations
            # This is harder to estimate from the sample, so we use the sample percentage
            no_cit_pct_all = no_citation_pct * (grand_totals.get('failed_claims', 0) / grand_totals.get('total_claims', 1)) if grand_totals.get('total_claims', 0) > 0 else 0
            summary.append(f"**Key Finding:** The most common claim failure mode is "
                          f"**No Citation Provided** ({format_percentage(no_citation_pct)} of sampled claims, "
                          f"estimated {format_percentage(no_cit_pct_all)} of all claims)")
        else:
            summary.append(f"**Key Finding:** The most common claim failure mode is "
                          f"**{top_claim_mode[1]['name']}** ({format_percentage(top_claim_mode[1]['percentage'])} of failed claims, "
                          f"{format_percentage(top_claim_pct)} of all claims)")
    
    if citations_report.get("category_statistics"):
        top_citation_mode = citations_report["category_statistics"][0]
        # Calculate percentage of ALL citations
        grand_totals = totals.get('grand_totals', {})
        total_citations_all = grand_totals.get('total_citations', 0)
        if total_citations_all > 0:
            # The percentage in the report is already of failed citations
            # We need to multiply by the failure rate to get percentage of all citations
            citation_failure_rate = grand_totals.get('citation_failure_rate', 0) / 100
            pct_of_all = top_citation_mode['percentage'] * citation_failure_rate
            summary.append(f"**Key Finding:** The most common citation failure mode is "
                          f"**{top_citation_mode['name']}** ({format_percentage(top_citation_mode['percentage'])} of failed citations, "
                          f"{format_percentage(pct_of_all)} of all citations)")
        else:
            summary.append(f"**Key Finding:** The most common citation failure mode is "
                          f"**{top_citation_mode['name']}** ({format_percentage(top_citation_mode['percentage'])} of failed citations)")
    
    summary.append("")
    return "\n".join(summary)


def generate_claims_analysis(report: Dict, totals: Dict = None) -> str:
    """Generate the unsupported claims analysis section."""
    sections = []
    sections.append("## Unsupported Claims Analysis\n")
    
    # Metadata
    if report.get("is_sample"):
        sections.append(f"*Analysis based on a stratified sample of {report.get('sample_size', 'N/A')} claims*\n")
    
    # Adjust for the fact that we sampled 50 per system (200 total) but only analyzed those with citations
    total_sampled = report.get('sample_size', 200)  # Total claims sampled
    total_analyzed = report.get('total_claims_classified', 0)  # Claims that had citations
    missing_citations = total_sampled - total_analyzed  # Claims without citations
    
    sections.append(f"**Total Claims Sampled:** {total_sampled}\n")
    sections.append(f"**Claims with Citations Analyzed:** {total_analyzed}\n")
    sections.append(f"**Claims without Citations:** {missing_citations}\n")
    sections.append(f"**Analysis Date:** {report.get('report_date', 'N/A')}\n")
    
    # Failure mode distribution table
    sections.append("### Failure Mode Distribution\n")
    
    if report.get("failure_mode_distribution"):
        headers = ["Failure Mode", "Description", "Count", "Percentage (of total sampled)"]
        rows = []
        
        # Calculate adjusted percentages based on total sampled
        total_sampled = report.get('sample_size', 200)
        missing_citations = total_sampled - total_analyzed
        
        # Add existing failure modes
        for code, data in sorted(report["failure_mode_distribution"].items(), 
                                 key=lambda x: x[1]["count"], reverse=True):
            adjusted_percentage = (data["count"] / total_sampled) * 100
            rows.append([
                data["name"],
                code.replace("_", " ").title(),
                data["count"],
                format_percentage(adjusted_percentage)
            ])
        
        # Add "No Citation Provided" for missing claims if there are any
        if missing_citations > 0:
            no_citation_percentage = (missing_citations / total_sampled) * 100
            rows.append([
                "No Citation Provided",
                "Claim had no citation",
                missing_citations,
                format_percentage(no_citation_percentage)
            ])
        
        # Sort rows by count (descending)
        rows.sort(key=lambda x: int(x[2]), reverse=True)
        
        sections.append(create_markdown_table(headers, rows))
        sections.append("")
    
    # System-level analysis
    if report.get("system_analysis"):
        sections.append("### System-Level Breakdown\n")
        
        headers = ["System", "Sampled", "With Citations", "No Citations", "Top Failure Mode", "Count"]
        rows = []
        
        # Calculate claims per system from total sampled divided by number of systems
        num_systems = len(report["system_analysis"])
        total_sampled = report.get('sample_size', 200)
        claims_per_system = total_sampled // num_systems if num_systems > 0 else 0
        
        for system, data in report["system_analysis"].items():
            system_total = data["total"]
            system_no_citations = claims_per_system - system_total
            
            if data.get("by_mode"):
                top_mode = max(data["by_mode"].items(), key=lambda x: x[1])
                mode_name = report["failure_mode_distribution"].get(top_mode[0], {}).get("name", top_mode[0])
                rows.append([
                    system.replace("_", " ").title(),
                    claims_per_system,
                    system_total,
                    system_no_citations,
                    mode_name,
                    top_mode[1]
                ])
            else:
                rows.append([
                    system.replace("_", " ").title(),
                    claims_per_system,
                    system_total,
                    system_no_citations,
                    "N/A",
                    0
                ])
        
        sections.append(create_markdown_table(headers, rows))
        sections.append("")
    
    return "\n".join(sections)


def generate_citations_analysis(report: Dict, totals: Dict = None) -> str:
    """Generate the citation failure analysis section."""
    sections = []
    sections.append("## Citation Failure Mode Analysis\n")
    
    sections.append(f"**Total Citations Analyzed:** {report.get('total_citations_analyzed', 0)}\n")
    sections.append(f"**Analysis Date:** {report.get('analysis_date', 'N/A')}\n")
    
    # Category statistics table
    sections.append("### Failure Mode Categories\n")
    
    if report.get("category_statistics"):
        headers = ["Category", "Description", "Count", "% of Failed Citations", "% of All Citations"]
        rows = []
        
        # Get total citation failure rate
        grand_totals = totals.get('grand_totals', {}) if totals else {}
        citation_failure_rate = grand_totals.get('citation_failure_rate', 0) / 100 if grand_totals else 0
        
        for category in report["category_statistics"]:
            pct_of_all = category["percentage"] * citation_failure_rate if citation_failure_rate > 0 else 0
            rows.append([
                category["name"],
                category["description"][:60] + "...",
                category["count"],
                format_percentage(category["percentage"]),
                format_percentage(pct_of_all)
            ])
        
        sections.append(create_markdown_table(headers, rows))
        sections.append("")
        
        # Detailed descriptions
        sections.append("### Detailed Category Descriptions\n")
        for category in report["category_statistics"]:
            sections.append(f"**{category['name']}** ({category['code']})")
            sections.append(f"- {category['description']}")
            sections.append("")
    
    # System statistics
    if report.get("system_statistics"):
        sections.append("### System Performance Comparison\n")
        
        headers = ["System", "Total Citations", "Top Failure Category", "Count"]
        rows = []
        
        for system, data in report["system_statistics"].items():
            rows.append([
                system.replace("_", " ").title(),
                data["total"],
                data.get("top_category", "N/A").replace("_", " ").title(),
                data["categories"].get(data.get("top_category", ""), 0)
            ])
        
        sections.append(create_markdown_table(headers, rows))
        sections.append("")
    
    return "\n".join(sections)


def generate_cross_system_analysis(claims_report: Dict, citations_report: Dict, totals: Dict = None) -> str:
    """Generate cross-system comparative analysis."""
    sections = []
    sections.append("## Cross-System Analysis\n")
    
    # Combine system data
    systems = set()
    if claims_report.get("system_analysis"):
        systems.update(claims_report["system_analysis"].keys())
    if citations_report.get("system_statistics"):
        systems.update(citations_report["system_statistics"].keys())
    
    if systems:
        sections.append("### Comparative Performance Metrics\n")
        
        headers = ["System", "Unsupported Claims", "Non-Supporting Citations", "Total Issues"]
        rows = []
        
        for system in sorted(systems):
            claims_count = claims_report.get("system_analysis", {}).get(system, {}).get("total", 0)
            citations_count = citations_report.get("system_statistics", {}).get(system, {}).get("total", 0)
            
            rows.append([
                system.replace("_", " ").title(),
                claims_count,
                citations_count,
                claims_count + citations_count
            ])
        
        rows.sort(key=lambda x: x[3])  # Sort by total issues
        sections.append(create_markdown_table(headers, rows))
        sections.append("")
        
        # Detailed error breakdown by system - Claims (Transposed)
        sections.append("### Claim Failure Rates by System\n")
        
        if claims_report.get("system_analysis") and claims_report.get("failure_mode_distribution"):
            # Calculate claims per system
            num_systems = len(claims_report["system_analysis"])
            total_sampled = claims_report.get('sample_size', 200)
            claims_per_system = total_sampled // num_systems if num_systems > 0 else 0
            
            # Get all failure modes
            all_modes = list(claims_report["failure_mode_distribution"].keys())
            
            # Create transposed table with failure modes as rows and systems as columns
            sorted_systems = sorted(systems)
            system_names = [s.replace("_", " ").title() for s in sorted_systems if s in claims_report.get("system_analysis", {})]
            headers = ["Failure Mode"] + system_names
            rows = []
            
            # Add "Total Sampled" row
            total_row = ["Total Sampled"]
            for system in sorted_systems:
                if system in claims_report.get("system_analysis", {}):
                    total_row.append(str(claims_per_system))
            rows.append(total_row)
            
            # Add row for each failure mode
            for mode in all_modes:
                mode_name = claims_report["failure_mode_distribution"][mode]["name"]
                row = [mode_name]
                
                for system in sorted_systems:
                    if system in claims_report.get("system_analysis", {}):
                        sys_data = claims_report["system_analysis"][system]
                        count = sys_data.get("by_mode", {}).get(mode, 0)
                        # Get system-specific totals if available
                        if totals and system in totals.get('systems', {}):
                            system_totals = totals['systems'][system]
                            system_total_claims = system_totals.get('total_claims', claims_per_system)
                            system_failure_rate = system_totals.get('claim_failure_rate', 0) / 100
                            # The count represents failures in the sample
                            # We need to estimate what percentage of ALL claims this represents
                            pct_of_sample_failures = count / sys_data['total'] if sys_data.get('total', 0) > 0 else 0
                            pct = system_failure_rate * pct_of_sample_failures * 100
                        else:
                            pct = (count / claims_per_system) * 100 if claims_per_system > 0 else 0
                        row.append(f"{count} ({pct:.1f}%)")
                
                rows.append(row)
            
            # Add "No Citation Provided" row
            no_cit_row = ["No Citation Provided"]
            for system in sorted_systems:
                if system in claims_report.get("system_analysis", {}):
                    sys_data = claims_report["system_analysis"][system]
                    system_total = sys_data.get("total", 0)
                    no_citations = claims_per_system - system_total
                    # Get system-specific totals if available
                    if totals and system in totals.get('systems', {}):
                        system_totals = totals['systems'][system]
                        system_total_claims = system_totals.get('total_claims', claims_per_system)
                        # Estimate the percentage of all claims without citations
                        # This is tricky because we don't have exact data
                        # We use the ratio in the sample as an estimate
                        no_cit_pct = (no_citations / claims_per_system) * 100
                    else:
                        no_cit_pct = (no_citations / claims_per_system) * 100 if claims_per_system > 0 else 0
                    no_cit_row.append(f"{no_citations} ({no_cit_pct:.1f}%)")
            rows.append(no_cit_row)
            
            sections.append(create_markdown_table(headers, rows))
            sections.append("")
        
        # Detailed error breakdown by system - Citations (Transposed)
        sections.append("### Citation Failure Rates by System\n")
        
        if citations_report.get("system_statistics") and citations_report.get("category_statistics"):
            # Get all categories
            all_categories = [cat["code"] for cat in citations_report["category_statistics"]]
            category_names = {cat["code"]: cat["name"] for cat in citations_report["category_statistics"]}
            
            # Create transposed table with categories as rows and systems as columns
            sorted_systems = sorted(systems)
            system_names = [s.replace("_", " ").title() for s in sorted_systems if s in citations_report.get("system_statistics", {})]
            headers = ["Citation Failure Mode"] + system_names
            rows = []
            
            # Add "Total Analyzed" row
            total_row = ["Total Analyzed"]
            for system in sorted_systems:
                if system in citations_report.get("system_statistics", {}):
                    sys_data = citations_report["system_statistics"][system]
                    total_row.append(str(sys_data.get("total", 0)))
            rows.append(total_row)
            
            # Add row for each category
            for cat in all_categories:
                cat_name = category_names[cat]
                row = [cat_name]
                
                for system in sorted_systems:
                    if system in citations_report.get("system_statistics", {}):
                        sys_data = citations_report["system_statistics"][system]
                        count = sys_data.get("categories", {}).get(cat, 0)
                        # Calculate percentage of ALL citations for this system
                        if totals and system in totals.get('systems', {}):
                            system_totals = totals['systems'][system]
                            system_total_citations = system_totals.get('total_citations', 1)
                            system_citation_failure_rate = system_totals.get('citation_failure_rate', 0) / 100
                            # The count represents failures in the analyzed sample
                            # Calculate what fraction of failed citations this category represents
                            pct_of_failures = count / sys_data['total'] if sys_data.get('total', 0) > 0 else 0
                            # Then multiply by the overall failure rate to get percentage of all citations
                            pct_of_all = system_citation_failure_rate * pct_of_failures * 100
                            row.append(f"{count} ({pct_of_all:.1f}%)")
                        elif sys_data.get("total", 0) > 0:
                            pct = (count / sys_data["total"]) * 100
                            row.append(f"{count} ({pct:.1f}%)")
                        else:
                            row.append("0 (0.0%)")
                
                rows.append(row)
            
            sections.append(create_markdown_table(headers, rows))
            sections.append("")
        
        # System strengths and weaknesses
        sections.append("### System Characteristics\n")
        
        for system in sorted(systems):
            system_name = system.replace("_", " ").title()
            sections.append(f"**{system_name}:**")
            
            # Claims analysis
            if system in claims_report.get("system_analysis", {}):
                claim_data = claims_report["system_analysis"][system]
                if claim_data.get("by_mode"):
                    top_claim_mode = max(claim_data["by_mode"].items(), key=lambda x: x[1])
                    mode_name = claims_report.get("failure_mode_distribution", {}).get(
                        top_claim_mode[0], {}
                    ).get("name", top_claim_mode[0])
                    sections.append(f"- Most common claim issue: {mode_name} ({top_claim_mode[1]} cases)")
            
            # Citations analysis
            if system in citations_report.get("system_statistics", {}):
                cite_data = citations_report["system_statistics"][system]
                top_cite = cite_data.get("top_category", "N/A").replace("_", " ").title()
                sections.append(f"- Most common citation issue: {top_cite}")
            
            sections.append("")
    
    return "\n".join(sections)


def generate_conclusions(claims_report: Dict, citations_report: Dict) -> str:
    """Generate conclusions and recommendations."""
    sections = []
    sections.append("## Conclusions and Recommendations\n")
    
    sections.append("### Key Findings\n")
    
    # Top issues across both analyses
    findings = []
    
    if claims_report.get("failure_mode_distribution"):
        top_modes = sorted(claims_report["failure_mode_distribution"].items(), 
                          key=lambda x: x[1]["count"], reverse=True)[:2]
        for mode, data in top_modes:
            findings.append(f"- **{data['name']}** accounts for {format_percentage(data['percentage'])} "
                          f"of unsupported claims")
    
    if citations_report.get("category_statistics"):
        top_cats = citations_report["category_statistics"][:2]
        for cat in top_cats:
            findings.append(f"- **{cat['name']}** represents {format_percentage(cat['percentage'])} "
                          f"of citation failures")
    
    sections.extend(findings)
    sections.append("")
    
    sections.append("### Recommendations\n")
    
    # Generate recommendations based on top failure modes
    recommendations = []
    
    # Check for specific failure patterns
    if citations_report.get("category_statistics"):
        top_cat = citations_report["category_statistics"][0]
        if top_cat["code"] == "insufficient_detail":
            recommendations.append("1. **Improve citation precision:** Systems should ensure citations "
                                 "contain specific evidence that directly supports claims")
        elif top_cat["code"] == "topic_mismatch":
            recommendations.append("1. **Enhance relevance filtering:** Better alignment needed between "
                                 "claim content and citation selection")
    
    if claims_report.get("failure_mode_distribution"):
        if "partial_support" in claims_report["failure_mode_distribution"]:
            if claims_report["failure_mode_distribution"]["partial_support"]["percentage"] > 50:
                recommendations.append("2. **Strengthen claim validation:** High rate of partially supported "
                                     "claims suggests need for more rigorous fact-checking")
    
    recommendations.append("3. **System-specific improvements:** Focus optimization efforts on systems "
                         "with highest error rates")
    recommendations.append("4. **Regular monitoring:** Implement continuous evaluation to track "
                         "improvement over time")
    
    sections.extend(recommendations)
    sections.append("")
    
    return "\n".join(sections)


def generate_markdown_report(
    claims_report_path: Path,
    citations_report_path: Path,
    totals_path: Path,
    output_path: Path
) -> None:
    """
    Generate a comprehensive markdown report from analysis JSON files.
    
    Args:
        claims_report_path: Path to failure_mode_analysis_report.json
        citations_report_path: Path to citation_failure_mode_analysis_report.json
        output_path: Path for the output markdown file
    """
    # Load reports
    claims_report = load_json_report(claims_report_path)
    citations_report = load_json_report(citations_report_path)
    totals = load_json_report(totals_path) if totals_path.exists() else {}
    
    # Build report sections
    report_sections = []
    
    # Header
    report_sections.append("# Citation and Claim Failure Mode Analysis Report")
    report_sections.append(f"\n*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
    report_sections.append("---\n")
    
    # Executive Summary
    report_sections.append(generate_executive_summary(claims_report, citations_report, totals))
    
    # Unsupported Claims Analysis
    report_sections.append(generate_claims_analysis(claims_report, totals))
    
    # Citation Failure Analysis
    report_sections.append(generate_citations_analysis(citations_report, totals))
    
    # Cross-System Analysis
    report_sections.append(generate_cross_system_analysis(claims_report, citations_report, totals))
    
    # Conclusions
    report_sections.append(generate_conclusions(claims_report, citations_report))
    
    # Footer
    report_sections.append("---")
    report_sections.append("*This report was automatically generated from citation evaluation data.*")
    
    # Write report
    full_report = "\n".join(report_sections)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(full_report)
    
    print(f"✓ Markdown report generated: {output_path}")


def main():
    """Main entry point for the script."""
    # Define paths relative to project root
    project_root = Path(__file__).parent.parent
    
    claims_report_path = project_root / "reports" / "failure_mode_analysis_report.json"
    citations_report_path = project_root / "reports" / "citation_failure_mode_analysis_report.json"
    totals_path = project_root / "data" / "statistics" / "totals.json"
    output_path = project_root / "reports" / "failure_mode_analysis.md"
    
    # Check input files exist
    if not claims_report_path.exists():
        raise FileNotFoundError(f"Claims report not found: {claims_report_path}")
    if not citations_report_path.exists():
        raise FileNotFoundError(f"Citations report not found: {citations_report_path}")
    
    # Generate report
    generate_markdown_report(claims_report_path, citations_report_path, totals_path, output_path)


if __name__ == "__main__":
    main()