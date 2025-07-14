# Deep Debate Implementation Roadmap

## Overview
Transform the current polite academic debate into passionate, thorough discussions with real stakes and progressive conflict escalation.

## Key Changes Required

### 1. **Enhanced Perspective Generation** (Stage 1 Upgrade)
**Current**: Generic roles like "心理学家", "社会学家"
**Enhanced**: Detailed professional identities with stakes

**Code Changes Needed**:
- Replace `_generate_perspectives_node()` prompt with deep perspective generator
- Add fields to GraphState: `professional_stakes`, `methodology_biases`, `past_commitments`
- Include conflict seeds and attack targets for each perspective

**Example Output**:
```json
{
  "name": "Dr. 李婉清",
  "title": "老年心理学专家，协和医院，30年临床经验",
  "stakes": "刚发表论文支持晚年再婚，申请$200万NIH基金中",
  "bias": "偏好纵向研究，不信任横断面调查",
  "will_attack": "对手的小样本研究和缺乏控制组"
}
```

### 2. **Progressive Conflict Escalation** (4-Round System)
**Current**: Static debate rounds with same intensity
**Enhanced**: Escalating conflict with different requirements per round

**Round Structure**:
- **Round 1**: Professional disagreement (cite studies, question methodology)
- **Round 2**: Methodological warfare (attack fundamental assumptions)
- **Round 3**: Stakes revelation (expose career/funding conflicts)
- **Round 4**: Forced collaboration (find pragmatic solutions under pressure)

**Code Changes**:
```python
# Add to _debate_round_node()
escalation_rules = {
    1: "professional_disagreement",
    2: "methodological_warfare", 
    3: "stakes_revelation",
    4: "forced_collaboration"
}
```

### 3. **Evidence Warfare System** (5-Tier Scoring)
**Current**: "Cite 1-2 sources per claim"
**Enhanced**: Detailed evidence quality scoring with attack mechanics

**Evidence Tiers**:
- **Tier 1 (5pts)**: Meta-analyses, large RCTs (n>1000)
- **Tier 2 (3pts)**: Single RCTs, longitudinal studies
- **Tier 3 (1pt)**: Case studies, expert opinions
- **Tier 0 (0pts)**: Unsupported claims

**Attack Multipliers**:
- Methodology flaw: -50% to evidence score
- Conflict of interest: -30%
- Sample bias: -40%

**Code Implementation**:
```python
def calculate_evidence_score(claims, attacks):
    base_score = sum(evidence_tier_scores)
    penalties = sum(attack_multipliers)
    bonuses = sum(defense_bonuses)
    return base_score * (1 - penalties) * (1 + bonuses)
```

### 4. **Enhanced Moderator with Conflict Enforcement**
**Current**: Simple decision between rounds/voting
**Enhanced**: Active conflict enforcement and evidence auditing

**New Moderator Functions**:
- Evidence quality auditing
- Conflict intensity monitoring
- Stake revelation enforcement
- Pressure injection for weak debates

**Code Changes**:
```python
# Add to _moderator_decision_node()
conflict_intensity = assess_conflict_level(debate_content)
evidence_scores = calculate_evidence_scores(perspectives)
if conflict_intensity < required_level:
    return inject_conflict_pressure()
```

### 5. **Pressure Element Integration**
**Current**: No external pressures
**Enhanced**: Real-world constraints that force decisions

**Pressure Types**:
- Deadline urgency ("Board meeting next Tuesday")
- Budget constraints ("Only $120万 available")
- Public accountability ("Results published in journal")
- Stakeholder pressure ("Families and institutions watching")

## Implementation Priority

### Phase 1: Foundation (Week 1)
1. **Update GraphState** to include new fields
2. **Enhance perspective generation** with detailed backgrounds
3. **Add evidence scoring system** framework

### Phase 2: Conflict Escalation (Week 2)
1. **Implement 4-round escalation** system
2. **Update perspective prompts** for each round type
3. **Add attack/defense mechanics** to evidence system

### Phase 3: Moderator Enhancement (Week 3)
1. **Upgrade moderator decision logic** with conflict enforcement
2. **Add evidence auditing** capabilities
3. **Implement pressure injection** mechanisms

### Phase 4: Integration & Testing (Week 4)
1. **Test escalation flow** end-to-end
2. **Fine-tune conflict intensity** triggers
3. **Optimize evidence warfare** balance

## Expected Outcomes

### Before (Current System):
- Polite academic discussion
- Surface-level evidence citations
- Predictable perspective roles
- No personal stakes or passion

### After (Enhanced System):
- Passionate professional conflict
- Deep methodological warfare
- Personal/career stakes revealed
- Progressive escalation to practical solutions
- Real-world pressure for decisions

## Sample Deep Debate Flow

**Topic**: "65岁以上女性再婚的心理健康影响"

**Round 1**: Dr. 李婉清 (老年心理学家) cites 3 longitudinal studies supporting late-life remarriage, while Professor 张明华 (社会心理学家) attacks her sample selection bias and questions the 5-year follow-up period.

**Round 2**: Escalates to methodological warfare - 李婉清 attacks 张明华's reliance on questionnaire-based studies vs. her clinical assessments, while 张明华 exposes how 李婉清's positive bias stems from her therapy practice.

**Round 3**: Stakes revealed - 李婉清's $200万 NIH grant application depends on positive outcomes, while 张明华's 5-year research program would be invalidated if wrong. Past public statements are brought up.

**Round 4**: External pressure applied - nursing home association needs recommendation by Friday, budget is limited to $120万, and results will be published. Both experts forced to find pragmatic middle ground despite fundamental disagreements.

This creates the depth and passion of real workplace debates while maintaining intellectual rigor.