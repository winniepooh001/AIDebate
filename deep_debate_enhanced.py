"""
Enhanced Deep Debate System
Creates passionate, thorough debates with personal stakes and progressive conflict escalation
"""

import json
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

class DeepDebateEnhancer:
    """Enhances debate system with deeper conflict and more realistic passion"""
    
    def generate_deep_perspectives(self, topic: str, language: str = "中文") -> Dict[str, Any]:
        """Generate perspectives with specific expertise, stakes, and conflicts"""
        
        prompt = f"""
**Role**: Senior Debate Architect with 20+ years experience designing high-stakes academic/corporate debates

**Task**: Create a DEEP, passionate debate framework for: "{topic}"

**Requirements**: Design 3-4 perspectives that will create genuine intellectual conflict, not polite disagreement.

**Each Perspective Must Include**:
1. **Specific Professional Identity**: 
   - Exact role, institution, years of experience
   - Specific research/work they've published
   - Current projects/grants at stake

2. **Personal Stakes**: 
   - Career consequences if they're wrong
   - Reputation/funding/relationships at risk
   - Past public positions they must defend

3. **Methodological Bias**:
   - Preferred research methods (qualitative vs quantitative)
   - Theoretical framework they're committed to
   - Data types they trust/distrust

4. **Conflict Seeds**:
   - Specific studies/approaches they'll attack
   - Methodological weaknesses they'll exploit
   - Personal grudges or professional rivalries

**Output Format (JSON)**:
{{
  "perspectives": [
    {{
      "name": "Dr. 李明华",
      "title": "临床心理学家，北京大学医学院，25年经验",
      "expertise": "专攻老年抑郁症治疗，发表论文85篇，主编《老年心理健康指南》",
      "current_stakes": "刚获得$500万NIH资助研究晚年生活质量，如果立场错误将影响5年研究方向",
      "methodology_bias": "偏好大样本定量研究，不信任小样本定性研究",
      "core_position": "基于统计数据支持65岁以上女性再婚的心理健康益处",
      "will_attack": "质疑对手的样本选择偏差和缺乏长期跟踪数据",
      "personal_motivation": "职业生涯建立在积极老龄化理论上，不能承认晚年重大变化有害",
      "past_commitments": "2019年在《Nature》发表社论支持老年人生活方式灵活性"
    }}
  ],
  "evidence_requirements": {{
    "tier1_required": ["Meta分析", "大型纵向研究(n>1000)", "多中心RCT"],
    "tier2_accepted": ["观察性研究", "队列研究", "专家共识"],
    "tier3_weak": ["案例研究", "横断面调查"],
    "attack_targets": ["样本偏差", "研究方法缺陷", "利益冲突"]
  }},
  "escalation_framework": {{
    "round1": "专业分歧 - 引用具体研究，质疑对手方法论",
    "round2": "理论冲突 - 挑战核心假设，暴露方法偏见",
    "round3": "利益揭露 - 揭示个人/职业利益，攻击动机",
    "round4": "务实妥协 - 在时间/资源压力下寻找可行方案"
  }},
  "pressure_elements": {{
    "deadline": "董事会下周二要最终建议",
    "budget_constraint": "只有120万预算用于老年服务项目",
    "public_accountability": "结果将在《中国老年医学杂志》发表",
    "stakeholder_pressure": "养老院协会和家属代表都在关注"
  }}
}}

请用{language}回答，确保每个观点都有真实的专业冲突和个人利益牵扯。
        """
        
        return {
            "prompt": prompt,
            "expected_structure": "Deep perspectives with stakes, conflicts, and escalation framework"
        }
    
    def create_evidence_warfare_prompt(self, perspective_name: str, perspective_data: Dict, 
                                     round_num: int, opposing_claims: List[str], 
                                     topic: str, language: str = "中文") -> str:
        """Create prompts that force deep evidence-based conflict"""
        
        # Get escalation level
        escalation_levels = {
            1: "professional_disagreement",
            2: "methodological_warfare", 
            3: "stakes_revelation",
            4: "forced_collaboration"
        }
        
        current_level = escalation_levels.get(round_num, "professional_disagreement")
        
        if current_level == "professional_disagreement":
            conflict_instruction = """
**Round 1 - Professional Disagreement**:
- Cite specific studies by EXACT name, year, journal, author
- Question opponent's methodology with specific technical critiques
- Defend your position with tier-1 evidence
- Acknowledge 1 limitation of your own evidence (to show intellectual honesty)
"""
        elif current_level == "methodological_warfare":
            conflict_instruction = """
**Round 2 - Methodological Warfare**:
- Attack the FUNDAMENTAL assumptions underlying opponent's research approach
- Expose sampling biases, measurement issues, or analytical flaws
- Escalate to paradigm-level disagreements (quantitative vs qualitative)
- Reveal how opponent's methodology serves their career interests
"""
        elif current_level == "stakes_revelation":
            conflict_instruction = """
**Round 3 - Stakes & Motivation Exposure**:
- Reveal the career/financial stakes behind opponent's position
- Point out conflicts of interest in their research funding
- Show how their past public statements lock them into this position
- Demonstrate urgency due to external pressures (deadlines, budgets)
"""
        else:  # forced_collaboration
            conflict_instruction = """
**Round 4 - Forced Pragmatic Collaboration**:
- Acknowledge opponent's legitimate concerns while maintaining core position
- Propose concrete compromises that address practical constraints
- Find middle ground that allows all parties to save face
- Focus on actionable next steps given real-world limitations
"""
        
        prompt = f"""
**Your Professional Identity**: {perspective_data.get('name', 'Expert')}
**Title**: {perspective_data.get('title', 'Professional')}
**Background**: {perspective_data.get('expertise', 'Experienced professional')}

**Current Situation**: 
Topic: {topic}
Your Stakes: {perspective_data.get('current_stakes', 'Professional reputation')}
Your Bias: {perspective_data.get('methodology_bias', 'Standard methods')}
Must Defend: {perspective_data.get('past_commitments', 'Previous positions')}

**Opponent Claims to Attack**:
{chr(10).join([f"- {claim}" for claim in opposing_claims])}

{conflict_instruction}

**Evidence Requirements**:
- Every major claim needs specific study citation (Author, Year, Journal)
- Attack opponent's evidence quality using: sample size, methodology, replication failures
- Defend against expected attacks on your evidence
- Show how the stakes/pressure affect your analysis

**Personal Motivation**:
{perspective_data.get('personal_motivation', 'Professional commitment to evidence-based practice')}

**Specific Targets to Attack**:
{perspective_data.get('will_attack', 'Methodological weaknesses in opposing research')}

**Required Output Structure**:
### 核心立场 (基于{perspective_data.get('core_position', '专业观点')}):
[2-3 specific, evidence-backed claims]

### 证据支持:
1. **主要研究**: [具体研究名称] (作者, 年份, 期刊) - [具体发现]
2. **支持数据**: [具体数据] - [为什么这个数据可信]
3. **我的证据局限**: [承认1个研究局限性，显示学术诚实]

### 对手攻击:
1. **[对手1]的问题**: [具体方法论缺陷]
2. **[对手2]的偏见**: [利益冲突或样本偏差]

### 压力因素:
[如何external pressure影响这个决策的紧迫性]

请用{language}回答，语气要体现{round_num}轮的激烈程度和个人利益牵扯。
        """
        
        return prompt
    
    def create_evidence_scoring_system(self) -> Dict[str, Any]:
        """Create detailed evidence quality scoring for moderator"""
        
        return {
            "evidence_tiers": {
                "tier1": {
                    "score": 5,
                    "types": ["Meta-analysis", "Systematic review", "Large RCT (n>500)"],
                    "requirements": ["Multiple studies", "High sample size", "Peer reviewed"]
                },
                "tier2": {
                    "score": 3,
                    "types": ["Single RCT", "Longitudinal study", "Large cohort"],
                    "requirements": ["Controlled design", "Adequate sample", "Published"]
                },
                "tier3": {
                    "score": 1,
                    "types": ["Case studies", "Expert opinion", "Small surveys"],
                    "requirements": ["Some documentation", "Professional source"]
                },
                "tier0": {
                    "score": 0,
                    "types": ["Unsupported claims", "Anecdotes", "Blog posts"],
                    "requirements": ["None"]
                }
            },
            "attack_multipliers": {
                "methodology_flaw": 0.5,  # Halves evidence score
                "conflict_of_interest": 0.7,  # Reduces by 30%
                "sample_bias": 0.6,  # Reduces by 40%
                "replication_failure": 0.3  # Reduces by 70%
            },
            "defense_bonuses": {
                "acknowledges_limitations": 1.2,  # 20% bonus for intellectual honesty
                "multiple_converging_studies": 1.5,  # 50% bonus for triangulation
                "addresses_criticisms": 1.3  # 30% bonus for thoughtful response
            }
        }
    
    def create_moderator_conflict_enforcer(self, round_num: int, max_rounds: int) -> str:
        """Create moderator prompt that enforces deeper conflict"""
        
        prompt = f"""
**Role**: Senior Academic Debate Moderator & Conflict Enforcer
**Experience**: 25+ years moderating high-stakes academic/corporate debates
**Mission**: Ensure DEEP intellectual conflict, not polite academic discussion

**Round {round_num}/{max_rounds} Analysis Requirements**:

## Evidence Quality Audit
For each perspective, calculate:
**Evidence Score** = Base Points + Bonuses - Penalties
- Tier 1 Evidence (Meta-analyses): 5 points each
- Tier 2 Evidence (RCTs): 3 points each  
- Tier 3 Evidence (Case studies): 1 point each
- Methodology attacks: -50% to target's score
- Acknowledged limitations: +20% bonus
- Multiple converging studies: +50% bonus

## Conflict Intensity Assessment
**Required Conflict Levels by Round**:
- Round 1: Methodological disagreement (cite specific studies)
- Round 2: Paradigm clash (quantitative vs qualitative warfare)
- Round 3: Stakes revelation (career/funding conflicts exposed)
- Round 4: Forced collaboration (external pressure for solutions)

## Decision Matrix:
- **ANOTHER_ROUND**: If conflict intensity < required level OR evidence gaps >2 points
- **ESCALATE_CONFLICT**: If participants are being too polite/academic
- **INTRODUCE_PRESSURE**: If no personal/professional stakes revealed
- **VOTE**: Only if intense conflict has been sustained AND evidence thoroughly examined

## Required Analysis Format:
**EVIDENCE SCORES**:
[Perspective 1]: [X]/10 points - [详细证据质量评估]
[Perspective 2]: [X]/10 points - [具体方法论优劣]

**CONFLICT INTENSITY**: [Low/Medium/High] 
- Personal stakes revealed: [Yes/No]
- Methodological warfare: [Present/Absent]  
- Career consequences discussed: [Yes/No]

**PRESSURE FACTORS**:
- Deadline urgency: [如何影响决策]
- Budget constraints: [如何限制选项]
- Public accountability: [如何增加风险]

**CONFLICT ENFORCEMENT**:
If conflict is too weak, inject:
- "Dr. X, your 2019 paper directly contradicts this position - explain the change"
- "The $2M funding you're seeking depends on this position being correct"
- "Your methodology has been criticized by [specific expert] - respond"

**DECISION**: [ANOTHER_ROUND|ESCALATE_CONFLICT|INTRODUCE_PRESSURE|VOTE]
**REASONING**: [Evidence quality gaps and conflict intensity analysis]
**NEXT_FOCUS**: [Specific methodological flaw or stake revelation required]
        """
        
        return prompt
    
    def suggest_implementation_changes(self) -> Dict[str, List[str]]:
        """Provide specific suggestions for code changes"""
        
        return {
            "perspective_generation_changes": [
                "Replace generic role assignment with detailed professional backgrounds",
                "Add stakes calculation (funding, reputation, career trajectory)",
                "Include past commitments that must be defended",
                "Add methodological biases and preferred evidence types",
                "Generate specific studies/approaches each perspective will attack"
            ],
            
            "round_progression_changes": [
                "Implement 4-tier escalation: Professional → Methodological → Stakes → Collaboration",
                "Add round-specific conflict requirements",
                "Include external pressure factors (deadlines, budgets, accountability)",
                "Force revelation of personal/career stakes in round 3",
                "Require pragmatic compromise in final round"
            ],
            
            "evidence_system_changes": [
                "Implement 5-tier evidence scoring system",
                "Add attack multipliers for methodology flaws",
                "Include defense bonuses for intellectual honesty",
                "Require specific study citations (Author, Year, Journal)",
                "Force acknowledgment of evidence limitations"
            ],
            
            "moderator_enhancement_changes": [
                "Add conflict intensity monitoring",
                "Implement evidence quality auditing",
                "Include stake revelation enforcement",
                "Add pressure injection mechanisms",
                "Create escalation triggers for weak conflicts"
            ],
            
            "prompt_engineering_changes": [
                "Add personal motivation sections to all prompts",
                "Include specific attack targets for each perspective",
                "Force acknowledgment of limitations and biases",
                "Add urgency/deadline pressure to decision making",
                "Include career consequence awareness in responses"
            ]
        }

# Example usage and testing
if __name__ == "__main__":
    enhancer = DeepDebateEnhancer()
    
    # Example deep perspective generation
    topic = "65岁以上女性再婚对心理健康的影响"
    deep_framework = enhancer.generate_deep_perspectives(topic)
    
    print("Deep Debate Framework Generated:")
    print(json.dumps(deep_framework, indent=2, ensure_ascii=False))
    
    # Example evidence warfare prompt
    perspective_data = {
        "name": "Dr. 李明华",
        "title": "临床心理学家，北京大学医学院，25年经验",
        "current_stakes": "刚获得$500万NIH资助研究晚年生活质量",
        "methodology_bias": "偏好大样本定量研究",
        "core_position": "支持65岁以上女性再婚的心理健康益处",
        "will_attack": "质疑对手的样本选择偏差",
        "personal_motivation": "职业生涯建立在积极老龄化理论上"
    }
    
    warfare_prompt = enhancer.create_evidence_warfare_prompt(
        "Dr. 李明华", perspective_data, 2, 
        ["晚年再婚增加心理压力", "老年人适应能力有限"], 
        topic
    )
    
    print("\nEvidence Warfare Prompt (Round 2):")
    print(warfare_prompt)