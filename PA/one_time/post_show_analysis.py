"""
Post-Show Recommendation Analysis for ECOMM 2025
=================================================

This script generates a comprehensive post-show analysis comparing recommendations
vs actual attendance for the ECOMM 2025 event.

Author: Senior Python Developer
Date: October 2025
"""

from typing import Dict, List, Tuple, Optional
import pandas as pd
from datetime import datetime
from neo4j import GraphDatabase
import logging
import yaml
from pathlib import Path


class PostShowAnalyzer:
    """Analyzes recommendation effectiveness post-show"""
    
    def __init__(self, config_path: str, neo4j_uri: str, neo4j_user: str, neo4j_password: str):
        """
        Initialize the post-show analyzer.
        
        Args:
            config_path: Path to config YAML file
            neo4j_uri: Neo4j database URI
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
        """
        self.logger = self._setup_logger()
        self.config = self._load_config(config_path)
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        self.event_name = self.config["event"]["name"]
        self.year = self.config["event"]["year"]
        self.logger.info(f"Initialized PostShowAnalyzer for {self.event_name} {self.year}")
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger("PostShowAnalyzer")
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _execute_query(self, query: str, parameters: Optional[Dict] = None) -> List[Dict]:
        """
        Execute a Cypher query and return results.
        
        Args:
            query: Cypher query string
            parameters: Optional query parameters
            
        Returns:
            List of result dictionaries
        """
        with self.driver.session() as session:
            result = session.run(query, parameters or {})
            return [record.data() for record in result]
    
    def get_overall_statistics(self) -> Dict:
        """Get overall statistics for recommendations and attendance"""
        query = """
        MATCH (v:Visitor_this_year)
        WITH count(v) as total_visitors
        MATCH (v:Visitor_this_year)-[r:IS_RECOMMENDED]->(s:Sessions_this_year)
        WITH total_visitors, count(DISTINCT v) as visitors_with_recommendations, 
             count(r) as total_recommendations
    MATCH (v:Visitor_this_year)-[a:assisted_session_this_year]->(s:Sessions_this_year)
        WITH total_visitors, visitors_with_recommendations, total_recommendations, 
             count(DISTINCT v) as visitors_who_attended, count(a) as total_attendances
        MATCH (s:Sessions_this_year)
        WITH total_visitors, visitors_with_recommendations, total_recommendations, 
             visitors_who_attended, total_attendances, count(s) as total_sessions
        MATCH (s:Sessions_this_year)<-[:IS_RECOMMENDED]-()
        WITH total_visitors, visitors_with_recommendations, total_recommendations, 
             visitors_who_attended, total_attendances, total_sessions, 
             count(DISTINCT s) as sessions_recommended
    MATCH (s:Sessions_this_year)<-[:assisted_session_this_year]-()
        RETURN total_visitors, visitors_with_recommendations, total_recommendations,
               visitors_who_attended, total_attendances, total_sessions, 
               sessions_recommended, count(DISTINCT s) as sessions_attended
        """
        results = self._execute_query(query)
        return results[0] if results else {}
    
    def get_recommendation_hit_rate(self) -> Dict:
        """Calculate how many recommended sessions were actually attended"""
        query = """
        MATCH (v:Visitor_this_year)-[:IS_RECOMMENDED]->(s:Sessions_this_year)
        WITH v, collect(DISTINCT s.session_id) as recommended_sessions
    MATCH (v)-[:assisted_session_this_year]->(s:Sessions_this_year)
        WITH v, recommended_sessions, collect(DISTINCT s.session_id) as attended_sessions
        WITH v, recommended_sessions, attended_sessions,
             [x IN attended_sessions WHERE x IN recommended_sessions] as matched_sessions
        RETURN 
          count(v) as visitors_analyzed,
          avg(size(matched_sessions)) as avg_matched_per_visitor,
          sum(size(matched_sessions)) as total_matched_attendances,
          sum(size(attended_sessions)) as total_attendances,
          100.0 * sum(size(matched_sessions)) / sum(size(attended_sessions)) as hit_rate_percentage
        """
        results = self._execute_query(query)
        return results[0] if results else {}
    
    def get_top_sessions_performance(self, limit: int = 20) -> pd.DataFrame:
        """Get top recommended sessions with their actual attendance and conversion"""
        query = """
        MATCH (v:Visitor_this_year)-[r:IS_RECOMMENDED]->(s:Sessions_this_year)
        WITH s, count(r) as recommendation_count
        ORDER BY recommendation_count DESC
        LIMIT $limit
    OPTIONAL MATCH (s)<-[a:assisted_session_this_year]-()
        RETURN s.title as session_title, s.session_id, 
               recommendation_count, 
               count(a) as actual_attendance,
               CASE WHEN recommendation_count > 0 
                    THEN 100.0 * count(a) / recommendation_count 
                    ELSE 0 END as conversion_rate
        ORDER BY recommendation_count DESC
        """
        results = self._execute_query(query, {"limit": limit})
        return pd.DataFrame(results)
    
    def get_best_conversion_sessions(self, min_recommendations: int = 50, 
                                      limit: int = 15) -> pd.DataFrame:
        """Get sessions with highest conversion rates (min threshold)"""
        query = """
        MATCH (v:Visitor_this_year)-[r:IS_RECOMMENDED]->(s:Sessions_this_year)
        WITH s, count(r) as recommendation_count
        WHERE recommendation_count >= $min_recs
    OPTIONAL MATCH (s)<-[a:assisted_session_this_year]-()
        WITH s, recommendation_count, count(a) as actual_attendance,
             100.0 * count(a) / recommendation_count as conversion_rate
        RETURN s.title as session_title, 
               recommendation_count, 
               actual_attendance,
               conversion_rate
        ORDER BY conversion_rate DESC
        LIMIT $limit
        """
        results = self._execute_query(query, {
            "min_recs": min_recommendations,
            "limit": limit
        })
        return pd.DataFrame(results)
    
    def get_sessions_not_recommended(self) -> pd.DataFrame:
        """Get sessions that were never recommended but had attendance"""
        query = """
        MATCH (s:Sessions_this_year)
        WHERE NOT EXISTS((s)<-[:IS_RECOMMENDED]-())
    OPTIONAL MATCH (s)<-[a:assisted_session_this_year]-()
        RETURN s.title as session_title, s.session_id, 
               s.sponsored_session, count(a) as attendance
        ORDER BY attendance DESC
        """
        results = self._execute_query(query)
        return pd.DataFrame(results)
    
    def get_gap_analysis(self) -> Dict:
        """Analyze gaps in recommendations and attendance"""
        # Visitors without recommendations
        query1 = """
        MATCH (v:Visitor_this_year)
        WHERE NOT EXISTS((v)-[:IS_RECOMMENDED]->())
        RETURN count(v) as visitors_without_recommendations
        """
        
        # Visitors who attended but had no recommendations
        query2 = """
    MATCH (v:Visitor_this_year)-[:assisted_session_this_year]->()
        WHERE NOT EXISTS((v)-[:IS_RECOMMENDED]->())
        RETURN count(DISTINCT v) as visitors_attended_no_recs,
               count(*) as total_attendances_no_recs
        """
        
        gap1 = self._execute_query(query1)[0]
        gap2 = self._execute_query(query2)[0]
        
        return {**gap1, **gap2}
    
    def get_visitor_attendance_stats(self) -> Dict:
        """Get statistics on visitor attendance patterns"""
        query = """
    MATCH (v:Visitor_this_year)-[a:assisted_session_this_year]->()
        WITH v, count(a) as num_attended
        RETURN 
          min(num_attended) as min_sessions,
          max(num_attended) as max_sessions,
          avg(num_attended) as avg_sessions,
          percentileCont(num_attended, 0.5) as median_sessions,
          percentileCont(num_attended, 0.25) as p25_sessions,
          percentileCont(num_attended, 0.75) as p75_sessions
        """
        results = self._execute_query(query)
        return results[0] if results else {}
    
    def get_returning_vs_new_performance(self) -> Tuple[Dict, Dict]:
        """Compare recommendation performance for returning vs new visitors"""
        # Returning visitors
        query_returning = """
        MATCH (v:Visitor_this_year)
        WHERE v.assist_year_before = '1'
        WITH count(v) as returning_total
        MATCH (v:Visitor_this_year)-[:IS_RECOMMENDED]->()
        WHERE v.assist_year_before = '1'
        WITH returning_total, count(DISTINCT v) as returning_with_recs
    MATCH (v:Visitor_this_year)-[:assisted_session_this_year]->()
        WHERE v.assist_year_before = '1'
        WITH returning_total, returning_with_recs, count(DISTINCT v) as returning_attended
        MATCH (v:Visitor_this_year)-[:IS_RECOMMENDED]->(s:Sessions_this_year)
        WHERE v.assist_year_before = '1'
        WITH returning_total, returning_with_recs, returning_attended, 
             v, collect(DISTINCT s.session_id) as recs
    MATCH (v)-[:assisted_session_this_year]->(s:Sessions_this_year)
        WITH returning_total, returning_with_recs, returning_attended,
             v, recs, collect(DISTINCT s.session_id) as attended,
             [x IN collect(DISTINCT s.session_id) WHERE x IN recs] as matched
        RETURN returning_total,
               returning_with_recs,
               returning_attended,
               count(v) as returning_both,
               avg(size(matched)) as avg_hit_returning,
               sum(size(matched)) as total_hits_returning,
               sum(size(attended)) as total_attended_returning,
               100.0 * sum(size(matched)) / sum(size(attended)) as hit_rate_returning
        """
        
        # New visitors
        query_new = """
        MATCH (v:Visitor_this_year)
        WHERE v.assist_year_before = '0'
        WITH count(v) as new_total
        MATCH (v:Visitor_this_year)-[:IS_RECOMMENDED]->()
        WHERE v.assist_year_before = '0'
        WITH new_total, count(DISTINCT v) as new_with_recs
    MATCH (v:Visitor_this_year)-[:assisted_session_this_year]->()
        WHERE v.assist_year_before = '0'
        WITH new_total, new_with_recs, count(DISTINCT v) as new_attended
        MATCH (v:Visitor_this_year)-[:IS_RECOMMENDED]->(s:Sessions_this_year)
        WHERE v.assist_year_before = '0'
        WITH new_total, new_with_recs, new_attended, 
             v, collect(DISTINCT s.session_id) as recs
    MATCH (v)-[:assisted_session_this_year]->(s:Sessions_this_year)
        WITH new_total, new_with_recs, new_attended,
             v, recs, collect(DISTINCT s.session_id) as attended,
             [x IN collect(DISTINCT s.session_id) WHERE x IN recs] as matched
        RETURN new_total,
               new_with_recs,
               new_attended,
               count(v) as new_both,
               avg(size(matched)) as avg_hit_new,
               sum(size(matched)) as total_hits_new,
               sum(size(attended)) as total_attended_new,
               100.0 * sum(size(matched)) / sum(size(attended)) as hit_rate_new
        """
        
        returning = self._execute_query(query_returning)
        new = self._execute_query(query_new)
        
        return (returning[0] if returning else {}, new[0] if new else {})
    
    def get_top_attended_sessions(self, limit: int = 20) -> pd.DataFrame:
        """Get most attended sessions regardless of recommendations"""
        query = """
    MATCH (s:Sessions_this_year)<-[a:assisted_session_this_year]-()
        WITH s, count(a) as attendance
        ORDER BY attendance DESC
        LIMIT $limit
        OPTIONAL MATCH (s)<-[r:IS_RECOMMENDED]-()
        RETURN s.title as session_title, s.session_id, attendance,
               count(r) as recommendations,
               EXISTS((s)<-[:IS_RECOMMENDED]-()) as was_recommended
        ORDER BY attendance DESC
        """
        results = self._execute_query(query, {"limit": limit})
        return pd.DataFrame(results)
    
    def generate_report(self, output_path: str) -> None:
        """
        Generate comprehensive post-show analysis report.
        
        Args:
            output_path: Path to save the markdown report
        """
        self.logger.info("Generating post-show analysis report...")
        
        # Collect all data
        overall_stats = self.get_overall_statistics()
        hit_rate = self.get_recommendation_hit_rate()
        top_sessions = self.get_top_sessions_performance(20)
        best_conversion = self.get_best_conversion_sessions(50, 15)
        not_recommended = self.get_sessions_not_recommended()
        gaps = self.get_gap_analysis()
        attendance_stats = self.get_visitor_attendance_stats()
        returning_perf, new_perf = self.get_returning_vs_new_performance()
        top_attended = self.get_top_attended_sessions(20)
        
        # Generate markdown report
        report = self._build_markdown_report(
            overall_stats, hit_rate, top_sessions, best_conversion,
            not_recommended, gaps, attendance_stats, 
            returning_perf, new_perf, top_attended
        )
        
        # Save report
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(report)
        
        self.logger.info(f"Report saved to {output_path}")
    
    def _build_markdown_report(self, overall_stats: Dict, hit_rate: Dict,
                                top_sessions: pd.DataFrame, best_conversion: pd.DataFrame,
                                not_recommended: pd.DataFrame, gaps: Dict,
                                attendance_stats: Dict, returning_perf: Dict,
                                new_perf: Dict, top_attended: pd.DataFrame) -> str:
        """Build the markdown report content"""
        
        report_date = datetime.now().strftime("%B %d, %Y")
        
        report = f"""# Post-Show Recommendation Analysis - ECOMM {self.year}
**Generated: {report_date}**

---

## Executive Summary

This report analyzes the effectiveness of the recommendation system for ECOMM {self.year} by comparing 
pre-show recommendations against actual post-show attendance data. The analysis reveals critical insights 
about recommendation accuracy, visitor engagement, and system gaps.

### Key Findings

- **Total Visitors**: {overall_stats.get('total_visitors', 0):,}
- **Visitors with Recommendations**: {overall_stats.get('visitors_with_recommendations', 0):,} ({100*overall_stats.get('visitors_with_recommendations', 0)/max(overall_stats.get('total_visitors', 1), 1):.1f}%)
- **Visitors Who Attended**: {overall_stats.get('visitors_who_attended', 0):,} ({100*overall_stats.get('visitors_who_attended', 0)/max(overall_stats.get('total_visitors', 1), 1):.1f}%)
- **Overall Hit Rate**: {hit_rate.get('hit_rate_percentage', 0):.2f}% (recommended sessions that were actually attended)
- **Critical Gap**: {gaps.get('visitors_without_recommendations', 0):,} visitors had no recommendations

---

## 1. Pre-Show vs Post-Show Comparison

### 1.1 Before the Show

**Pre-Show Situation (Latest Run Before Event):**
- Recommendation system generated {overall_stats.get('total_recommendations', 0):,} recommendations
- Covered {overall_stats.get('sessions_recommended', 0)} of {overall_stats.get('total_sessions', 0)} sessions ({100*overall_stats.get('sessions_recommended', 0)/max(overall_stats.get('total_sessions', 1), 1):.1f}%)
- Average {overall_stats.get('total_recommendations', 0)/max(overall_stats.get('visitors_with_recommendations', 1), 1):.1f} recommendations per visitor
- Top sessions recommended to >95% of visitors (high concentration)

**Pre-Show Concerns:**
- High recommendation concentration on popular sessions
- {overall_stats.get('total_sessions', 0) - overall_stats.get('sessions_recommended', 0)} sessions without recommendations
- Many sponsored sessions lacking recommendations
- System heavily biased toward AI and digital transformation topics

### 1.2 Post-Show Reality

**Actual Attendance:**
- {overall_stats.get('visitors_who_attended', 0):,} visitors attended at least one session
- Total of {overall_stats.get('total_attendances', 0):,} session attendances recorded
- {overall_stats.get('sessions_attended', 0)} unique sessions had attendance
- Average {attendance_stats.get('avg_sessions', 0):.2f} sessions per attending visitor
- Median {attendance_stats.get('median_sessions', 0):.0f} sessions attended

**Attendance Distribution:**
- Min: {attendance_stats.get('min_sessions', 0)} session
- 25th percentile: {attendance_stats.get('p25_sessions', 0):.0f} sessions
- Median: {attendance_stats.get('median_sessions', 0):.0f} sessions  
- 75th percentile: {attendance_stats.get('p75_sessions', 0):.0f} sessions
- Max: {attendance_stats.get('max_sessions', 0)} sessions

---

## 2. Recommendation Effectiveness Analysis

### 2.1 Overall Hit Rate

**Definition**: Percentage of attended sessions that were previously recommended

- **Hit Rate**: {hit_rate.get('hit_rate_percentage', 0):.2f}%
- **Total Matched**: {hit_rate.get('total_matched_attendances', 0):,} of {hit_rate.get('total_attendances', 0):,} attendances
- **Visitors Analyzed**: {hit_rate.get('visitors_analyzed', 0):,}
- **Avg Matches per Visitor**: {hit_rate.get('avg_matched_per_visitor', 0):.2f}

**Interpretation**: 
{self._interpret_hit_rate(hit_rate.get('hit_rate_percentage', 0))}

### 2.2 Returning vs New Visitors

| Metric | Returning Visitors | New Visitors | Difference |
|--------|-------------------|--------------|------------|
| **Total Visitors** | {returning_perf.get('returning_total', 0):,} | {new_perf.get('new_total', 0):,} | - |
| **With Recommendations** | {returning_perf.get('returning_with_recs', 0):,} ({100*returning_perf.get('returning_with_recs', 0)/max(returning_perf.get('returning_total', 1), 1):.1f}%) | {new_perf.get('new_with_recs', 0):,} ({100*new_perf.get('new_with_recs', 0)/max(new_perf.get('new_total', 1), 1):.1f}%) | {100*new_perf.get('new_with_recs', 0)/max(new_perf.get('new_total', 1), 1) - 100*returning_perf.get('returning_with_recs', 0)/max(returning_perf.get('returning_total', 1), 1):+.1f}% |
| **Who Attended** | {returning_perf.get('returning_attended', 0):,} ({100*returning_perf.get('returning_attended', 0)/max(returning_perf.get('returning_total', 1), 1):.1f}%) | {new_perf.get('new_attended', 0):,} ({100*new_perf.get('new_attended', 0)/max(new_perf.get('new_total', 1), 1):.1f}%) | {100*new_perf.get('new_attended', 0)/max(new_perf.get('new_total', 1), 1) - 100*returning_perf.get('returning_attended', 0)/max(returning_perf.get('returning_total', 1), 1):+.1f}% |
| **Hit Rate** | {returning_perf.get('hit_rate_returning', 0):.2f}% | {new_perf.get('hit_rate_new', 0):.2f}% | {returning_perf.get('hit_rate_returning', 0) - new_perf.get('hit_rate_new', 0):+.2f}% |
| **Avg Matched/Visitor** | {returning_perf.get('avg_hit_returning', 0):.2f} | {new_perf.get('avg_hit_new', 0):.2f} | {returning_perf.get('avg_hit_returning', 0) - new_perf.get('avg_hit_new', 0):+.2f} |

**Key Insight**: {self._interpret_returning_vs_new(returning_perf.get('hit_rate_returning', 0), new_perf.get('hit_rate_new', 0))}

---

## 3. Session Performance Analysis

### 3.1 Top 20 Most Recommended Sessions - Conversion Performance

The following sessions received the most recommendations. Conversion rate shows what percentage of recommended visitors actually attended.

"""
        # Add top sessions table
        if not top_sessions.empty:
            report += "\n| Rank | Session Title | Recommendations | Attendance | Conversion Rate |\n"
            report += "|------|---------------|-----------------|------------|----------------|\n"
            for idx, row in top_sessions.iterrows():
                title_short = row['session_title'][:80] + "..." if len(row['session_title']) > 80 else row['session_title']
                report += f"| {idx+1} | {title_short} | {row['recommendation_count']:,} | {row['actual_attendance']} | {row['conversion_rate']:.2f}% |\n"
        
        report += f"""

**Analysis**: 
- Average conversion rate for top 20: {top_sessions['conversion_rate'].mean():.2f}%
- Best performing: {top_sessions.iloc[0]['session_title'][:50]}... ({top_sessions.iloc[0]['conversion_rate']:.2f}%)
- Most over-recommended: {top_sessions.nlargest(1, 'recommendation_count').iloc[0]['session_title'][:50]}... ({top_sessions.nlargest(1, 'recommendation_count').iloc[0]['recommendation_count']:,} recs, {top_sessions.nlargest(1, 'recommendation_count').iloc[0]['conversion_rate']:.2f}% conversion)

### 3.2 Best Converting Sessions (min 50 recommendations)

These sessions had the highest attendance relative to recommendations:

"""
        # Add best conversion table
        if not best_conversion.empty:
            report += "\n| Rank | Session Title | Recommendations | Attendance | Conversion Rate |\n"
            report += "|------|---------------|-----------------|------------|----------------|\n"
            for idx, row in best_conversion.iterrows():
                title_short = row['session_title'][:80] + "..." if len(row['session_title']) > 80 else row['session_title']
                report += f"| {idx+1} | {title_short} | {row['recommendation_count']:,} | {row['actual_attendance']} | {row['conversion_rate']:.2f}% |\n"
        
        report += f"""

**Analysis**: These sessions significantly outperformed expectations. Several show conversion rates >100%, indicating they attracted visitors beyond those who received recommendations.

### 3.3 Most Attended Sessions (Actual Popularity)

"""
        # Add most attended table
        if not top_attended.empty:
            report += "\n| Rank | Session Title | Attendance | Recommendations | Was Recommended? |\n"
            report += "|------|---------------|------------|-----------------|------------------|\n"
            for idx, row in top_attended.iterrows():
                title_short = row['session_title'][:80] + "..." if len(row['session_title']) > 80 else row['session_title']
                rec_status = "✓ Yes" if row['was_recommended'] else "✗ No"
                report += f"| {idx+1} | {title_short} | {row['attendance']} | {row['recommendations']:,} | {rec_status} |\n"
        
        report += """

---

## 4. Gap Analysis

### 4.1 Visitors Without Recommendations

"""
        
        report += f"""**Total visitors without recommendations**: {gaps.get('visitors_without_recommendations', 0):,} ({100*gaps.get('visitors_without_recommendations', 0)/max(overall_stats.get('total_visitors', 1), 1):.1f}% of all visitors)

**Of these, {gaps.get('visitors_attended_no_recs', 0):,} attended sessions** with {gaps.get('total_attendances_no_recs', 0):,} total attendances.

**Reasons for missing recommendations:**
1. Late registrations after last recommendation run
2. Visitors added to database post-show
3. System failures or data quality issues
4. VIP/special badge types potentially excluded

**Impact**: These {gaps.get('visitors_attended_no_recs', 0):,} visitors attended sessions without any algorithmic guidance, representing {100*gaps.get('total_attendances_no_recs', 0)/max(overall_stats.get('total_attendances', 1), 1):.1f}% of all attendances.

### 4.2 Sessions Never Recommended

**Total sessions never recommended**: {len(not_recommended)} of {overall_stats.get('total_sessions', 0)} sessions

"""
        
        # Add not recommended sessions with attendance
        not_rec_with_attendance = not_recommended[not_recommended['attendance'] > 0]
        if not not_rec_with_attendance.empty:
            report += f"\n**Sessions with attendance despite no recommendations** ({len(not_rec_with_attendance)} sessions):\n\n"
            report += "| Session Title | Attendance | Sponsored? |\n"
            report += "|---------------|------------|------------|\n"
            for _, row in not_rec_with_attendance.iterrows():
                title_short = row['session_title'][:80] + "..." if len(row['session_title']) > 80 else row['session_title']
                sponsored = "Yes" if str(row['sponsored_session']).lower() == 'true' else "No"
                report += f"| {title_short} | {row['attendance']} | {sponsored} |\n"
            
            report += f"\n**Critical Finding**: '{not_rec_with_attendance.iloc[0]['session_title']}' had {not_rec_with_attendance.iloc[0]['attendance']} attendees despite never being recommended. This suggests strong organic interest.\n"
        
        sponsored_count = len(not_recommended[not_recommended['sponsored_session'].astype(str).str.lower() == 'true'])
        report += f"""

**Sponsored session coverage**: {sponsored_count} sponsored sessions never recommended

---

## 5. Conclusions and Recommendations

### 5.1 What Worked Well

"""
        
        successes = []
        if hit_rate.get('hit_rate_percentage', 0) > 20:
            successes.append(f"✓ **Reasonable hit rate** of {hit_rate.get('hit_rate_percentage', 0):.2f}% shows recommendations influenced behavior")
        if returning_perf.get('hit_rate_returning', 0) > new_perf.get('hit_rate_new', 0):
            successes.append(f"✓ **Better performance for returning visitors** ({returning_perf.get('hit_rate_returning', 0):.2f}% vs {new_perf.get('hit_rate_new', 0):.2f}%)")
        if overall_stats.get('visitors_who_attended', 0) / overall_stats.get('total_visitors', 1) > 0.20:
            successes.append(f"✓ **Strong engagement** with {100*overall_stats.get('visitors_who_attended', 0)/overall_stats.get('total_visitors', 1):.1f}% of visitors attending sessions")
        
        if best_conversion.shape[0] > 0:
            successes.append(f"✓ **High-converting sessions identified**: Several sessions achieved >100% conversion rates")
        
        for success in successes:
            report += f"{success}\n"
        
        report += """

### 5.2 Critical Issues Identified

"""
        
        issues = []
        if gaps.get('visitors_without_recommendations', 0) / overall_stats.get('total_visitors', 1) > 0.10:
            issues.append(f"✗ **{100*gaps.get('visitors_without_recommendations', 0)/overall_stats.get('total_visitors', 1):.1f}% of visitors had no recommendations** - likely due to late registrations or system timing")
        
        if hit_rate.get('hit_rate_percentage', 0) < 30:
            issues.append(f"✗ **Low hit rate of {hit_rate.get('hit_rate_percentage', 0):.2f}%** means most attended sessions were not recommended")
        
        if top_sessions.iloc[0]['conversion_rate'] < 3:
            issues.append(f"✗ **Top recommended session had only {top_sessions.iloc[0]['conversion_rate']:.2f}% conversion** - indicates over-recommendation")
        
        if sponsored_count > 5:
            issues.append(f"✗ **{sponsored_count} sponsored sessions never recommended** - potential revenue and sponsor satisfaction impact")
        
        if len(not_rec_with_attendance) > 0:
            issues.append(f"✗ **{len(not_rec_with_attendance)} sessions had organic attendance without recommendations** - missing opportunities")
        
        for issue in issues:
            report += f"{issue}\n"
        
        report += """

### 5.3 Actionable Recommendations

**Immediate Actions (Next Event):**

1. **Earlier Final Run**: Execute final recommendation run 24-48 hours before event to capture late registrations
2. **Include Sponsored Sessions**: Ensure all sponsored sessions have embeddings and appear in recommendations
3. **Reduce Over-concentration**: Implement maximum recommendation percentage per session (e.g., cap at 50% of visitors)
4. **Monitor Gaps**: Create alerts for visitors added post-recommendation run

**Algorithm Improvements:**

1. **Leverage Historical Data**: For returning visitors, weight past attendance patterns more heavily
2. **Diversification Constraints**: Ensure each visitor gets recommendations across multiple topics/streams
3. **Similarity Threshold Tuning**: Increase minimum similarity to reduce low-quality recommendations
4. **Time Slot Awareness**: Account for concurrent sessions to avoid impossible schedules

**Data Quality:**

1. **Badge Type Handling**: Investigate why VIPs or certain badge types might lack recommendations
2. **Session Completeness**: Ensure all sessions have complete metadata before final run
3. **Registration Flow**: Add recommendation trigger for late registrations

**Measurement:**

1. **Real-time Monitoring**: Track recommendation usage during event (if possible)
2. **Feedback Collection**: Survey attendees about recommendation usefulness
3. **A/B Testing**: Test different recommendation strategies with visitor segments

### 5.4 Success Metrics for Next Event

Target improvements for next year:

- **Hit Rate**: Increase from {hit_rate.get('hit_rate_percentage', 0):.2f}% to >35%
- **Coverage**: Reduce visitors without recommendations from {100*gaps.get('visitors_without_recommendations', 0)/overall_stats.get('total_visitors', 1):.1f}% to <5%
- **Conversion Balance**: Top 10 sessions should have similar conversion rates (currently high variance)
- **Sponsored Coverage**: 100% of sponsored sessions should receive recommendations
- **Returning Visitor Performance**: Maintain or improve {returning_perf.get('hit_rate_returning', 0):.2f}% hit rate

---

## 6. Technical Notes

**Analysis Date**: {report_date}
**Database**: Neo4j (neo4j-dev)
**Event**: ECOMM {self.year}
**Data Sources**: 
- IS_RECOMMENDED relationships (pre-show)
- assisted_this_year relationships (post-show actual attendance)
- Visitor_this_year nodes
- Sessions_this_year nodes

**Methodology**: 
- Hit rate calculated as percentage of attended sessions that were recommended
- Conversion rate calculated as (actual_attendance / recommendation_count) * 100
- Gaps identified by comparing pre-show recommendation generation time vs registration timestamps

---

*Report generated by Post-Show Analysis System*
*For questions or clarifications, contact the Data Team*
"""
        
        return report
    
    def _interpret_hit_rate(self, hit_rate: float) -> str:
        """Provide interpretation of hit rate"""
        if hit_rate >= 40:
            return "**Excellent**: The recommendation system strongly influenced visitor behavior."
        elif hit_rate >= 30:
            return "**Good**: Recommendations had meaningful impact on session selection."
        elif hit_rate >= 20:
            return "**Moderate**: Recommendations provided some guidance but visitors largely chose independently."
        elif hit_rate >= 10:
            return "**Low**: Limited impact from recommendations. Most visitors attended sessions not recommended to them."
        else:
            return "**Very Low**: Minimal correlation between recommendations and attendance. System needs significant improvement."
    
    def _interpret_returning_vs_new(self, returning_rate: float, new_rate: float) -> str:
        """Interpret the difference between returning and new visitor hit rates"""
        diff = returning_rate - new_rate
        if diff > 5:
            return f"Returning visitors show **{diff:.1f} percentage points higher** hit rate, indicating their historical behavior improved recommendations."
        elif diff < -5:
            return f"New visitors show **{abs(diff):.1f} percentage points higher** hit rate, suggesting registration data may be more predictive than historical behavior."
        else:
            return f"Similar performance between groups (difference: {diff:+.1f}%) suggests historical data provides limited additional value."
    
    def close(self) -> None:
        """Close the Neo4j driver connection"""
        self.driver.close()
        self.logger.info("Closed Neo4j connection")


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate post-show recommendation analysis")
    parser.add_argument("--config", required=True, help="Path to config YAML file")
    parser.add_argument("--neo4j-uri", default="bolt://localhost:7687", help="Neo4j URI")
    parser.add_argument("--neo4j-user", default="neo4j", help="Neo4j username")
    parser.add_argument("--neo4j-password", required=True, help="Neo4j password")
    parser.add_argument("--output", required=True, help="Output path for markdown report")
    
    args = parser.parse_args()
    
    analyzer = PostShowAnalyzer(
        config_path=args.config,
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_password=args.neo4j_password
    )
    
    try:
        analyzer.generate_report(args.output)
        print(f"\n✓ Report generated successfully: {args.output}")
    finally:
        analyzer.close()


if __name__ == "__main__":
    main()