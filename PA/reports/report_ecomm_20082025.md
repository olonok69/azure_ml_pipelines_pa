# Comprehensive Analysis of Visitor Attributes and Recommendation System for Neo4j
**Updated Report - Current Database Statistics**

## Executive Summary

The Neo4j database contains **2,974 visitors** for this year's event, with **23.8% returning visitors** from last year's ecommerce shows. The recommendation system uses 6 key visitor attributes with 100% fill rate to generate personalized session recommendations. The system has successfully generated **59,496 recommendations** (average 20 per visitor) for **97 of 115 available sessions** (84.4% coverage), though recommendations show high concentration on popular sessions with the top session recommended to 98.1% of visitors.

## 1. Visitor Demographics and Retention

### 1.1 Overall Visitor Statistics
- **Total Visitors**: 2,974 (↑ 20.2% from previous 2,475)
- **Returning Visitors**: 708 (23.8%) (↑ from 539, 21.8%)
- **New Visitors**: 2,331 (78.4%) (↑ from 1,936, 78.2%)

### 1.2 Returning Visitor Analysis
**All 708 returning visitors attended ecommerce-related events last year:**
- **444 visitors** attended Ecomm (main event) - 62.7% of returning visitors
- **234 visitors** attended TFM London (Technology for Marketing) - 33.1% of returning visitors
- **35 visitors** attended BOTH events (4.9% of returning visitors)

**Key Insights:**
- Retention rate increased slightly to 23.8%
- 100% of returning visitors are from ecommerce events (consistent with previous)
- Core loyal audience growing but still represents minority of attendees
- Cross-event attendees (35) represent highly engaged segment

## 2. Data Completeness and Quality

### Overall Statistics
- **Total Recommendations Generated**: 59,496 (↑ 20.2% from 49,516)
- **Sessions Available**: 115 (unchanged)
- **Sessions with Recommendations**: 97 (84.3%)
- **Sessions Never Recommended**: 18 (15.7%)
- **Attribute Fill Rate**: 100% for all 6 attributes

### Attribute Overview

| Attribute | Unique Values | Fill Rate | NA Values | NA % |
|-----------|--------------|-----------|-----------|------|
| Country | 83 | 100% | 0 | 0% |
| what_is_your_industry | 23 | 100% | 208 | 6.99% |
| what_is_the_main_solution_you_are_most_excited_to_discover_at_the_show | 24 | 100% | 206 | 6.93% |
| what_best_describes_what_you_do | 17 | 100% | 207 | 6.96% |
| what_is_your_job_role | 6 | 100% | 205 | 6.89% |
| why_are_you_attending | 6 | 100% | 2 | 0.07% |

## 3. Detailed Attribute Analysis

### 3.1 Job Role Distribution
**Most Concentrated Attribute** (Only 6 unique values)

| Role | Count | Percentage | Change from Previous |
|------|-------|------------|---------------------|
| Manager | 839 | 28.21% | ↑ from 699 (28.24%) |
| Director/Head of Department | 823 | 27.67% | ↑ from 697 (28.16%) |
| C-level/Owner | 656 | 22.06% | ↑ from 555 (22.42%) |
| Executive/Assistant | 243 | 8.17% | ↑ from 166 (6.71%) |
| Executive | 208 | 6.99% | ↑ from 188 (7.60%) |
| NA | 205 | 6.89% | ↑ from 170 (6.87%) |

**Key Insights:**
- Leadership-heavy audience: 77.94% are Manager level or above
- Decision-maker concentration suitable for B2B focus
- Executive/Assistant role increased significantly (+46.4%)

### 3.2 What Best Describes What You Do
**17 unique categories**

Top 10 Categories:
| Activity | Count | Percentage | Change |
|----------|-------|------------|--------|
| General Marketing, Brand Marketing | 443 | 14.90% | ↑ from 344 (13.90%) |
| General Management | 350 | 11.77% | ↑ from 299 (12.08%) |
| Digital Marketing (SEO, PPC, Automations) | 344 | 11.57% | ↑ from 283 (11.43%) |
| Business Development | 323 | 10.86% | ↑ from 289 (11.68%) |
| Content Marketing, PR, Advertising | 261 | 8.78% | ↑ from 190 (7.68%) |
| eCommerce CX & Personalisation | 212 | 7.13% | ↑ from 184 (7.43%) |
| NA | 207 | 6.96% | ↑ from 172 (6.95%) |
| B2B Commerce | 162 | 5.45% | ↑ from 139 (5.62%) |
| Online Sales Optimisation for Retail | 139 | 4.67% | ↑ from 123 (4.97%) |
| IT | 127 | 4.27% | ↑ from 104 (4.20%) |

**Key Insights:**
- Marketing-focused audience (46.02% in marketing roles)
- Strong eCommerce specialization alignment
- Content Marketing showing significant growth (+37.4%)

### 3.3 Industry Distribution
**Most Diverse Attribute** (23 categories)

Top 10 Industries:
| Industry | Count | Percentage | Change |
|----------|-------|------------|--------|
| Fashion, Clothing & Accessories | 282 | 9.48% | ↑ from 237 (9.58%) |
| IT & Electronics | 250 | 8.41% | ↑ from 200 (8.08%) |
| Health, Beauty & Wellness | 246 | 8.27% | ↑ from 203 (8.20%) |
| Other | 223 | 7.50% | ↑ from 189 (7.64%) |
| NA | 208 | 6.99% | ↑ from 172 (6.95%) |
| Marketing & Advertising | 203 | 6.82% | ↑ from 166 (6.71%) |
| Manufacturing & Wholesale | 196 | 6.59% | ↑ from 165 (6.67%) |
| Home & Garden & Lifestyle | 188 | 6.32% | ↑ from 167 (6.75%) |
| Food & Beverage | 169 | 5.68% | ↑ from 143 (5.78%) |
| Finance, Banking & Insurance | 143 | 4.81% | ↑ from 120 (4.85%) |

### 3.4 Main Solution Excited to Discover
**AI Dominance Continues** (24 unique values)

| Solution | Count | Percentage | Change |
|----------|-------|------------|--------|
| Artificial Intelligence (AI) | 719 | 24.18% | ↑ from 606 (24.48%) |
| Marketing Content Creation & Management | 254 | 8.54% | ↑ from 188 (7.60%) |
| Lead Generation | 223 | 7.50% | ↑ from 179 (7.23%) |
| Marketing Analytics & SEO Tools | 220 | 7.40% | ↑ from 174 (7.03%) |
| NA | 206 | 6.93% | ↑ from 173 (6.99%) |
| Customer Acquisition | 174 | 5.85% | ↑ from 148 (5.98%) |
| CX & Personalisation | 127 | 4.27% | ↑ from 114 (4.61%) |
| Content Creation & Management | 126 | 4.24% | ↑ from 103 (4.16%) |
| Customer Engagement | 125 | 4.20% | ↑ from 104 (4.20%) |
| CRM | 124 | 4.17% | ↑ from 104 (4.20%) |

**Key Insights:**
- AI remains the dominant interest (1 in 4 visitors)
- Marketing Content Creation showing strong growth (+35.1%)
- Marketing technology solutions are primary focus

### 3.5 Country Distribution
**UK-Dominated** (83 countries represented, ↑ from 79)

| Country | Count | Percentage | Change |
|---------|-------|------------|--------|
| UK | 2,624 | 88.23% | ↑ from 2,174 (87.84%) |
| India | 28 | 0.94% | ↑ from 25 (1.01%) |
| USA | 25 | 0.84% | ↑ from 21 (0.85%) |
| Ghana | 14 | 0.47% | ↑ from 13 (0.53%) |
| Germany | 13 | 0.44% | ↑ from 12 (0.48%) |
| Portugal | 11 | 0.37% | New in top 10 |
| Turkey | 11 | 0.37% | New in top 10 |
| Pakistan | 11 | 0.37% | New in top 10 |
| Algeria | 11 | 0.37% | New in top 10 |
| The Netherlands | 10 | 0.34% | New in top 10 |

### 3.6 Why Are You Attending
**Clearest Intent** (Only 6 categories)

| Reason | Count | Percentage | Change |
|--------|-------|------------|--------|
| To optimise my organisation's online sales | 1,572 | 52.86% | ↑ from 1,393 (56.28%) |
| To improve our own in-house marketing | 1,193 | 40.11% | ↑ from 908 (36.69%) |
| Other | 202 | 6.79% | ↑ from 168 (6.79%) |
| To help others improve their marketing | 4 | 0.13% | ↑ from 3 (0.12%) |
| NA | 2 | 0.07% | = 2 (0.08%) |
| To advise/assist other organisations | 1 | 0.03% | = 1 (0.04%) |

## 4. Session Recommendation Analysis

### 4.1 Recommendation Coverage
- **Total Sessions**: 115 (unchanged)
- **Sessions Recommended**: 97 (84.3%)
- **Sessions Never Recommended**: 18 (15.7%)
- **Total Recommendations Made**: 59,496 (↑ 20.2% from 49,516)
- **Average per Visitor**: 20.01 sessions (unchanged)
- **Average per Session**: 613.4 recommendations (↑ from 510.5)

### 4.2 Top 10 Most Recommended Sessions

| Rank | Session Title | Times Recommended | % of Visitors | Date | Stage |
|------|--------------|-------------------|---------------|------|-------|
| 1 | Where eCommerce is headed in 2025 | 2,918 | 98.12% | Sept 25 | Future of Digital Commerce |
| 2 | The AI revolution: eCommerce in 2025 & beyond | 2,880 | 96.84% | Sept 24 | Future of Digital Commerce |
| 3 | Personal, purposeful, convenient | 2,806 | 94.35% | Sept 24 | Future of Digital Commerce |
| 4 | 25 years on the digital tails | 2,691 | 90.48% | Sept 24 | Global Growth & Fulfilment |
| 5 | The future of shopping: Live commerce | 2,562 | 86.15% | Sept 24 | Future of Digital Commerce |
| 6 | Made-to-measure bespoke platform | 2,512 | 84.47% | Sept 24 | Future of Digital Commerce |
| 7 | The new rules of conversion | 2,172 | 73.03% | Sept 24 | Optimisation Stage |
| 8 | David Bowie & retail media | 2,155 | 72.46% | Sept 25 | Future of Customer Experience |
| 9 | AI-driven transformation | 2,080 | 69.94% | Sept 24 | Future of Customer Experience |
| 10 | Next gen sustainability | 1,993 | 67.01% | Sept 25 | Future of Digital Commerce |

**Key Observations:**
- **Extreme concentration persists**: Top 3 sessions recommended to >94% of visitors
- **AI and future trends dominate**: 8/10 focus on future/AI/digital transformation
- **Day 1 heavy**: 7 sessions on Sept 24, 3 on Sept 25
- **Stage concentration**: 60% on Future of Digital Commerce Stage

### 4.3 Sessions Never Recommended (18 total)

**Breakdown by Type:**
- **Sponsored Sessions**: 10 (55.6%)
  - PayPal, Algolia, Anicca, Descartes, Merx, Insider, Nosto, Payabl, Orderwise Forterro, MAPP
- **Social/Networking Events**: 7 (38.9%)
  - Book signing and "Hang out" sessions
- **Regular Content Session**: 1 (5.5%)
  - "Building responsible AI: Governance, licensing and content integrity"

**Key Insights:**
- Sponsored sessions systematically excluded (likely missing embeddings)
- Social events appropriately excluded from content recommendations
- AI governance session surprisingly not recommended despite AI interest

### 4.4 Similarity Score Distribution

| Metric | Score |
|--------|-------|
| Minimum | 0.3318 |
| Q1 (25th percentile) | 0.6800 |
| Median | 0.6908 |
| Q3 (75th percentile) | 0.7118 |
| Maximum | 0.8189 |
| Average | 0.6939 |

## 5. Key Changes from Previous Report

### 5.1 Growth Metrics
- **Visitor Growth**: +20.2% (from 2,475 to 2,974)
- **Returning Visitor Growth**: +31.4% (from 539 to 708)
- **Recommendation Growth**: +20.2% (from 49,516 to 59,496)
- **Country Diversity**: +5% (from 79 to 83 countries)

### 5.2 Shifting Trends
- **In-house marketing focus increased**: 40.1% vs 36.7% previously
- **Executive/Assistant roles growing**: 8.2% vs 6.7% previously
- **Content Marketing surge**: +37.4% growth
- **Marketing Content Creation interest**: +35.1% growth

### 5.3 Persistent Patterns
- **AI dominance stable**: ~24% interested in AI solutions
- **UK concentration consistent**: ~88% of visitors
- **Recommendation concentration unchanged**: Top session still at 98%
- **Leadership audience maintained**: ~78% manager level or above

## 6. System Performance Issues and Opportunities

### 6.1 Recommendation Concentration Problem
**Issue**: Top session recommended to 98.12% of visitors
- Indicates over-reliance on popular sessions as fallbacks
- Reduces personalization effectiveness
- May overwhelm popular sessions with attendance

**Potential Causes:**
1. Low similarity thresholds (minimum 0.33)
2. Insufficient weight on differentiating attributes
3. Missing embeddings for specialized content

### 6.2 Coverage Gaps
**18 sessions (15.7%) never recommended:**
- 10 sponsored sessions lack recommendations
- Potential lost monetization/sponsor value
- AI governance session missing despite high AI interest

### 6.3 Visitor Segmentation Opportunities
**Underutilized segments:**
- 35 cross-event attendees (most engaged segment)
- 708 returning visitors (loyalty segment)
- 350 international visitors (11.77% non-UK)

## 7. Recommendations for Improvement

### 7.1 Immediate Actions
1. **Reduce fallback recommendations**: Increase minimum similarity threshold from 0.33 to 0.5
2. **Include sponsored sessions**: Generate embeddings for sponsored content
3. **Leverage returning visitor data**: Use last year's session attendance for better predictions
4. **Diversify recommendations**: Cap any single session at 50% of visitors

### 7.2 Data Enhancement
1. **Capture session attendance**: Track which recommended sessions visitors actually attend
2. **Add visitor segments**: Flag returning visitors, cross-event attendees, international
3. **Reduce NA values**: Implement better form validation (currently 6-7% NA)
4. **Expand categories**: "Other" represents 7.5% in Industry field

### 7.3 Algorithm Improvements
1. **Weight by entropy**: Give higher weight to high-entropy attributes (Industry, Solution Interest)
2. **Use historical behavior**: For 708 returning visitors, use past session attendance
3. **Implement diversity constraints**: Ensure recommendations span different topics/stages
4. **Time-slot awareness**: Avoid recommending concurrent sessions

### 7.4 Business Opportunities
1. **Retention programs**: Target 78.4% new visitors for next year retention
2. **Cross-event promotion**: Leverage 35 multi-event attendees as ambassadors
3. **Sponsored session integration**: Better integrate sponsor content into recommendations
4. **International expansion**: Growing international audience (11.8% vs 12.2% previously)

## 8. Conclusion

The recommendation system demonstrates continued growth with a 20.2% increase in visitors and maintained technical performance with reasonable similarity scores (median 0.69). However, the extreme concentration of recommendations (top session at 98.12%) indicates persistent over-reliance on popular content as fallbacks, reducing personalization value.

Key strengths include:
- Strong visitor growth (+20.2%) with improved retention (+31.4% returning visitors)
- Complete data coverage and clear visitor intent alignment
- Growing interest in content marketing and marketing technology
- Increased international diversity (83 countries vs 79)

Priority improvements should focus on:
1. Reducing recommendation concentration through higher similarity thresholds
2. Leveraging returning visitor history (708 visitors with past behavior data)
3. Including sponsored sessions in recommendations
4. Implementing diversity constraints to ensure varied content exposure

The system shows positive growth trends but needs refinement to deliver truly personalized experiences that justify the data collection effort and provide value to both the growing visitor base and sponsors.