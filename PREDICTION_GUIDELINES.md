# FPL Prediction Training Data Guidelines

## 🚨 **IMPORTANT: Exclude GW0 (Friendlies) from Training Data**

### **Background**
GW0 typically contains pre-season friendly matches, which are not representative of official Premier League performance. These should be excluded from all prediction models to ensure accuracy.

### **Updated Training Data Requirements**

#### **For Any Gameweek N Prediction:**
- ❌ **Don't Use**: GW0 (friendlies)
- ✅ **Use**: GW1 to GW(N-1) (official EPL matches only)

#### **Specific Examples:**

| **Prediction Target** | **Training Data** | **Minimum Required** |
|----------------------|-------------------|---------------------|
| GW2                  | GW1               | 1 gameweek          |
| GW3                  | GW1-GW2           | 2 gameweeks         |
| GW4                  | GW1-GW3           | 3 gameweeks         |
| GW5                  | GW1-GW4           | 4 gameweeks         |
| GW6+                 | GW1-GW(N-1)       | N-1 gameweeks       |

### **Code Implementation Updates**

#### **1. Update Gameweek Range Discovery**
```python
# OLD (includes friendlies)
for gw_num in range(0, 39):  # GW0 to GW38

# NEW (official matches only)  
for gw_num in range(1, 39):  # GW1 to GW38 (exclude GW0 friendlies)
```

#### **2. Update Training Data Requirements**
```python
# OLD (includes friendlies)
self.required_training_gws = list(range(0, target_gw))

# NEW (official matches only)
self.required_training_gws = list(range(1, target_gw))  # Start from GW1
```

#### **3. Update Documentation and Messages**
```python
print("📝 Note: GW0 (friendlies) excluded - using only official EPL matches")
print("Required for GWX prediction: GW1-GW(X-1) (official EPL only)")
```

### **Benefits of Excluding GW0**

1. **🎯 Better Accuracy**: Official match conditions vs friendly scenarios
2. **📊 Realistic Performance**: True competitive intensity and tactics
3. **🏆 Official Scoring**: Consistent refereeing and FPL point allocation
4. **🔄 Squad Rotation**: Real team selection vs experimental lineups
5. **⚽ Match Importance**: Competitive stakes vs pre-season preparation

### **Implementation Checklist**

- [ ] Update `discover_available_gameweeks()` to start from GW1
- [ ] Modify training data requirements to exclude GW0
- [ ] Update minimum gameweek requirements (GW2 needs 1 GW, GW3 needs 2 GWs, etc.)
- [ ] Add documentation notes about GW0 exclusion
- [ ] Update error messages to reflect official EPL matches only
- [ ] Test with actual data to ensure no GW0 contamination

### **Files to Update**

1. **`fpl_gw2_prediction.ipynb`** - Current notebook
2. **`fpl_gw5_prediction.py`** - ✅ Already updated
3. **Any future prediction scripts** - Follow this pattern
4. **`scripts/split_by_gameweek.py`** - Consider separate handling for GW0
5. **Documentation** - Update README and guides

### **Data Quality Impact**

| **Aspect** | **With GW0** | **Without GW0** |
|------------|---------------|-----------------|
| **Match Quality** | Mixed (friendlies + official) | Official EPL only |
| **Player Motivation** | Variable | Consistent |
| **Team Tactics** | Experimental | Competitive |
| **Squad Selection** | Rotation heavy | Strategic |
| **Prediction Accuracy** | Lower | Higher ✅ |

---

**Remember**: The goal is to predict official EPL performance, so training data should reflect official EPL conditions!